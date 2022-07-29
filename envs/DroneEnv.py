import airsim
import time

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DroneEnv(gym.Env):
    def __init__(self, env_path="envs/data/1.csv", sparse_reward=False):
        self.TIMESTEP_LIMIT = 100
        self.M = 4 # rows X
        self.N = 4 # cols Y
        self.LENGTH = 5
        self.TARGET_DIST = 2
        self.sparse_reward = sparse_reward
        self.timesteps = 0
        self.reward = 0.0

        self.observation_shape_depth = (1, 144, 256)
        self.observation_shape_scene = (3, 144, 256,)
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=self.observation_shape_scene,
                                                dtype=np.uint8)
    
        self.action_space = gym.spaces.Box(low=-1.0, 
                                           high=1.0, 
                                           shape=(2,), dtype=np.float32)

        self.target_pos = np.array([0, 2])
        self.target_pos_ = np.array([self.target_pos[0] * self.LENGTH + self.LENGTH/2, 
                                     self.target_pos[1] * self.LENGTH + self.LENGTH/2])
        self.target = np.array([self.target_pos[0] * self.LENGTH, 
                                self.target_pos[1] * self.LENGTH])

        self.process_map(env_path)
        # self.draw_map()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()


    def process_map(self, env_path):
        max_len = self.M * self.N
        self.map = np.empty((self.M, self.N, 4), dtype=np.bool)

        df = pd.read_csv(env_path)
        df = df[['Y', 'X', 'NorthWall?', 'EastWall?', 'SouthWall?', 'WestWall?']]
        
        for row in df.to_numpy():
            self.map[row[0], row[1]] = row[2:]
        
        self.rewards_map = np.zeros((self.M, self.N))
        self.recursive_reward(self.target_pos, max_len)
        self.rewards_map -= self.rewards_map.min()
        self.rewards_map /= self.rewards_map.max()



    def recursive_reward(self, pos, val):
        self.rewards_map[pos[0], pos[1]] = val
        n, e, s, w = self.map[pos[0], pos[1]]

        if not n and self.rewards_map[pos[0]+1, pos[1]] == 0:
            self.recursive_reward([pos[0]+1, pos[1]], val-1)
        if not e and self.rewards_map[pos[0], pos[1]+1] == 0:
            self.recursive_reward([pos[0], pos[1]+1], val-1)
        if not s and self.rewards_map[pos[0]-1, pos[1]] == 0:
            self.recursive_reward([pos[0]-1, pos[1]], val-1)
        if not w and self.rewards_map[pos[0], pos[1]-1] == 0:
            self.recursive_reward([pos[0], pos[1]-1], val-1)
        

    def draw_map(self, pos=None):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        for i in range(self.M):
            for j in range(self.N):
                x = self.LENGTH * i + self.LENGTH / 2
                y = self.LENGTH * j + self.LENGTH / 2
                
                if self.map[i, j, 0]:
                    ax.plot([x + self.LENGTH/2, x + self.LENGTH/2], 
                            [y - self.LENGTH/2, y + self.LENGTH/2], 'r')
                if self.map[i, j, 1]:
                    ax.plot([x - self.LENGTH/2, x + self.LENGTH/2], 
                            [y + self.LENGTH/2, y + self.LENGTH/2], 'r')
                if self.map[i, j, 2]:
                    ax.plot([x - self.LENGTH/2, x - self.LENGTH/2], 
                            [y - self.LENGTH/2, y + self.LENGTH/2], 'r')
                if self.map[i, j, 3]:
                    ax.plot([x - self.LENGTH/2, x + self.LENGTH/2], 
                            [y - self.LENGTH/2, y - self.LENGTH/2], 'r')

                ax.text(x, y, "{:.2f}".format(self.rewards_map[i, j]), fontsize=10)

        if pos:
            ax.scatter(pos[0], pos[1], c='b')

        ax.scatter(self.target_pos_[0], self.target_pos_[1], c='g')
        c = plt.Circle((self.target_pos_[0], self.target_pos_[1]), 
                    self.TARGET_DIST, color='g', fill=False)
        ax.add_artist(c)
        
        plt.show()
        

    def get_depth_img_(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.DepthVis, True, False),])
        depth = np.array(responses[0].image_data_float)
        depth = np.array(depth * 255.0, dtype=np.uint8)
        if (len(depth) != (144*256)):
            print('The depth camera returned bad data.')
            depth = np.ones((1, 144, 256)).astype(np.float32)
        else:
            depth = depth.reshape(1, responses[0].height, responses[0].width)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        return depth

    
    def get_scene_img_(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),])[0]
        img = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        if (len(img) != (144*256*3)):
            print('The scene camera returned bad data.')
            img = np.ones((3, 144, 256)).astype(np.float32)
        else:
            img = img.reshape(response.height, response.width, 3)
            img = np.moveaxis(img, -1, 0)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        return img


    def reset(self):
        # super().reset()
        self.timesteps = 0
        self.reward = 0.0
        self.pos = np.array([0.0, 0.0])
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToPositionAsync(0, 0, -1, 1).join()

        self.state = self.get_scene_img_()
        return self.state


    def dist(self, pos_x, pos_y, x, y):
        x_ = x * self.LENGTH + self.LENGTH / 2
        y_ = y * self.LENGTH + self.LENGTH / 2
        return np.sqrt((pos_x - x_)**2 + (pos_y - y_)**2)


    def get_reward(self, pos):
        pos_x = pos[0] + self.LENGTH/2
        pos_y = pos[1] + self.LENGTH/2

        x = int(max(pos_x // self.LENGTH, 0))
        y = int(max(pos_y // self.LENGTH, 0))
    
        r1 = self.rewards_map[x, y]
        d1 = self.dist(pos_x, pos_y, x, y)
        r2 = 0
        d2 = 1

        min_d = np.inf
        n, e, s, w = self.map[x, y]
        if not n:
            d2 = self.dist(pos_x, pos_y, x+1, y)
            if d2 < min_d:
                min_d = d2
                r2 = self.rewards_map[x+1, y]
        if not e:
            d2 = self.dist(pos_x, pos_y, x, y+1)
            if d2 < min_d:
                min_d = d2
                r2 = self.rewards_map[x, y+1]
        if not s:
            d2 = self.dist(pos_x, pos_y, x-1, y)
            if d2 < min_d:
                min_d = d2
                r2 = self.rewards_map[x-1, y]
        if not w:
            d2 = self.dist(pos_x, pos_y, x, y-1)
            if d2 < min_d:
                min_d = d2
                r2 = self.rewards_map[x, y-1]

        return 1 - self.LENGTH * (r1*d2 + r2*d1) / np.power(d2 + d1, 2)

    
    def episode(self):
        return self.timesteps, self.reward
    

    def step(self, actions):
        # axtionos : tuple of two elements
        #   0: velocity [-1, 1] -> [-2, 2] in unreal env
        #   1: angle [-1, 1] -> [-90, 90] in unreal env
        self.timesteps += 1

        vel = actions[0] * 2.0
        ang = actions[1] * 90

        # self.pos += np.array([vel, 0.0])
        self.client.rotateByYawRateAsync(ang, 0.5).join()
        self.client.moveByVelocityBodyFrameAsync(vel, 0, 0, 0.5).join()
        # time.sleep(10)

        self.state = self.get_scene_img_()
        # 'position': <Vector3r> {   
        #     'x_val': 2.1757102012634277,
        #     'y_val': 4.236437689542072e-08,
        #     'z_val': -0.4433230459690094}},
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        pos = np.array([position.x_val, position.y_val])

        if not self.sparse_reward:
            reward = self.get_reward(pos)
        else:
            reward = 0

        if self.client.simGetCollisionInfo().has_collided:
            self.reward += -self.TIMESTEP_LIMIT
            return self.state, -self.TIMESTEP_LIMIT, True, {}

        if self.timesteps > self.TIMESTEP_LIMIT:
            self.reward += -1 * reward
            return self.state, -1 * reward, True, {'truncated':True}

        self.client.moveToPositionAsync(position.x_val, position.y_val, -1, 1).join()

        if np.square(pos - self.target).sum() < self.TARGET_DIST:
            self.reward += self.TIMESTEP_LIMIT
            return self.state, 100, self.TIMESTEP_LIMIT, {'x': position.x_val, 
                                                           'y': position.y_val, 
                                                           'z': position.z_val}

        # # return self.state, reward, False, {}
        self.reward += -1 * reward
        return self.state, -1 * reward, False, {'x': position.x_val, 
                                                'y': position.y_val, 
                                                'z': position.z_val}
        