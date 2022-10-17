import airsim
import time

from PIL import Image
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch

class DroneEnv(gym.Env):
    def __init__(self, env_path="envs/data/2.csv", sparse_reward=False):
        self.TIMESTEP_LIMIT = 500
        self.M = 4 # rows X
        self.N = 4 # cols Y
        self.LENGTH = 5
        self.TARGET_DIST = 2
        self.sparse_reward = sparse_reward
        self.timesteps = 0
        self.reward = 0.0
        self.last_reward = 0.0

        self.observation_shape_depth = (1, 144, 256)
        self.observation_shape_scene = (3, 144, 256,)
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=self.observation_shape_scene,
                                                dtype=np.uint8)
    
        self.action_space = gym.spaces.Box(low=-1.0, 
                                           high=1.0, 
                                           shape=(1,), dtype=np.float32)

        self.target_pos = np.array([0, 1])
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
        
        self.rewards_map = np.ones((self.M, self.N))
        self.recursive_reward(self.target_pos, 0)
        self.rewards_map -= self.rewards_map.min()
        self.rewards_map /= self.rewards_map.max()


    def recursive_reward(self, pos, val):
        reduction = self.LENGTH
        self.rewards_map[pos[0], pos[1]] = val
        n, e, s, w = self.map[pos[0], pos[1]]

        if not n and self.rewards_map[pos[0]+1, pos[1]] == 1:
            self.recursive_reward([pos[0]+1, pos[1]], val-reduction)
        if not e and self.rewards_map[pos[0], pos[1]+1] == 1:
            self.recursive_reward([pos[0], pos[1]+1], val-reduction)
        if not s and self.rewards_map[pos[0]-1, pos[1]] == 1:
            self.recursive_reward([pos[0]-1, pos[1]], val-reduction)
        if not w and self.rewards_map[pos[0], pos[1]-1] == 1:
            self.recursive_reward([pos[0], pos[1]-1], val-reduction)
        

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
            depth = np.ones((144, 256, 1)).astype(np.float32)
        else:
            depth = depth.reshape(responses[0].height, responses[0].width, 1)
            img = img.astype(np.float32) / 255.0
            # img = np.expand_dims(img, axis=0)
        return depth

    
    def get_scene_img_(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),])[0]
        img = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        if (len(img) != (144*256*3)):
            print('The scene camera returned bad data.')
            img = np.ones((144, 256, 3)).astype(np.float32)
        else:
            img = img.reshape(response.height, response.width, 3)
            # img = np.moveaxis(img, -1, 0)
            img = img.astype(np.float32) / 255.0
            # img = np.expand_dims(img, axis=0)
        return img

    
    def random_move(self):
        moves = [0] * 5 + [1] * 2 + [0] * 8 + [-1] * 2 + [0] * 4 + [-1] * 2 + [0] * 6 + [1] * 2 + [0] * 6 + [1] * 3
        i = np.random.randint(len(moves))

        for m in moves[:i]:
            _ = self.step([m])


    def reset(self):
        # super().reset()
        self.timesteps = 0
        self.reward = 0.0
        self.pos = np.array([0.0, 0.0])
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToPositionAsync(0, 0, -1, 1).join()

        # self.random_move()

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

        reward = self.LENGTH * (r1*min_d + r2*d1) / np.power(min_d + d1, 2)

        # if d1 >= min_d:
        #     reward = r2 - min_d
        # else:
        #     reward = r1 - d1
        return reward
    

    def step(self, actions):
        # axtionos : tuple of two elements
        #   0: velocity [-1, 1] -> [-2, 2] in unreal env
        #   1: angle [-1, 1] -> [-90, 90] in unreal env
        self.timesteps += 1

        vel = 2.0
        ang = actions[0] * 900
        speed = 0.05

        # self.pos += np.array([vel, 0.0])
        drone_state = self.client.getMultirotorState()
        z = drone_state.kinematics_estimated.position.z_val + 0.5

        self.client.rotateByYawRateAsync(ang, speed).join()
        self.client.moveByVelocityBodyFrameAsync(vel, 0, -3*z, speed).join()
        # time.sleep(10)

        self.state = self.get_scene_img_()
        # s = torch.tensor(self.state[0].copy())
        # save_image(s, f'imgs/{self.timesteps}.png')

        # 'position': <Vector3r> {   
        #     'x_val': 2.1757102012634277,
        #     'y_val': 4.236437689542072e-08,
        #     'z_val': -0.4433230459690094}},
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        # self.client.moveToZAsync(-1, 1).join()
        pos = np.array([position.x_val, position.y_val])

        if not self.sparse_reward:
            reward = self.get_reward(pos)
        else:
            reward = 0

        # reward = (new_reward - self.last_reward) * 100
        # self.last_reward = new_reward

        # print(reward)
        if self.client.simGetCollisionInfo().has_collided:
            self.reward += -1
            return self.state, -1, True, {'episode':{
                                            'length': self.timesteps,
                                            'reward': self.reward
                                            }}

        if self.timesteps > self.TIMESTEP_LIMIT:
            self.reward += reward
            return self.state, reward, True, {'truncated':True, 
                                              'episode':{
                                                'length': self.timesteps,
                                                'reward': self.reward
                                              }}

        if np.square(pos - self.target).sum() < self.TARGET_DIST:
            self.reward += self.TIMESTEP_LIMIT
            self.last_reward = 0.0
            return self.state, self.TIMESTEP_LIMIT, True, {'episode':{
                                                                'length': self.timesteps,
                                                                'reward': self.reward
                                                            }}
        self.reward += reward
        return self.state, reward, False, {}
        