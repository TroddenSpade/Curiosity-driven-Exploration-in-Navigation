import airsim
import time

import gym
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        self.max_vel = 1.0
        self.min_vel = -1.0
        self.max_ang = 1.0
        self.min_ang = -1.0

        self.observation_shape_depth = (144, 256, 1)
        self.observation_shape_scene = (144, 256, 3)
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=255,
                                                shape=self.observation_shape_depth,
                                                dtype=np.uint8)
    
        self.action_space = gym.spaces.Box(low=-1.0, 
                                           high=1.0, 
                                           shape=(2,), dtype=np.float32)

        self.target_pos = np.array([0.0, 10.0])
        self.coords = np.load('./envs/coords1.npy')

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.TIMESTEP_LIMIT = 100
        self.timesteps = 0


    def get_depth_img_(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest(1, airsim.ImageType.DepthVis, True, False),])
        depth = np.array(responses[0].image_data_float)
        depth = np.array(depth * 255.0, dtype=np.uint8)
        if (len(depth) != (144*256)):
            print('The depth camera returned bad data.')
            depth = np.ones((144,256,1))
        else:
            depth = depth.reshape(responses[0].height, responses[0].width, 1)
        return depth

    
    def get_scene_img_(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),])[0]
        img = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        if (len(img) != (144*256*3)):
            print('The depth camera returned bad data.')
            img = np.ones((144,256,3))
        else:
            img = img.reshape(response.height, response.width, 3)
        # img = np.flipud(img)
        return img


    def reset(self):
        # super().reset()
        self.timesteps = 0
        self.pos = np.array([0.0, 0.0])
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToPositionAsync(0, 0, -1, 1).join()

        self.state = self.get_depth_img_()
        return self.state


    def get_scale(self, pos):
        idx = np.argmin(np.square(self.coords - pos).sum(axis=1))
        return idx/len(self.coords)

    
    def step(self, actions):
        # axtionos : tuple of two elements
        #   0: velocity [-1, 1] -> [0, 2] in unreal env
        #   1: angle [-1, 1] -> [-90, 90] in unreal env

        self.timesteps += 1

        vel = actions[0] + 1.0
        ang = actions[1] * 90

        # self.pos += np.array([vel, 0.0])
        self.client.rotateByYawRateAsync(ang, 0.5).join()
        self.client.moveByVelocityBodyFrameAsync(vel, 0, 0, 0.5).join()
        # self.client.moveByVelocityAsync(vel, 0, 0, 0.5).join()
        # time.sleep(0.1)

        self.state = self.get_depth_img_()

        # 'position': <Vector3r> {   
        #     'x_val': 2.1757102012634277,
        #     'y_val': 4.236437689542072e-08,
        #     'z_val': -0.4433230459690094}},
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        pos = np.array([position.x_val, position.y_val])
        scale = self.get_scale(pos)
        print(scale)

        if self.client.simGetCollisionInfo().has_collided:
            return self.state, -10, True, {}

        self.client.moveToPositionAsync(position.x_val, position.y_val, -1, 1).join()

        if self.timesteps > self.TIMESTEP_LIMIT:
            return self.state, scale, True, {'truncated':True}

        if np.square(pos - self.target_pos).sum() < 1:
            return self.state, +100, True, {'x': position.x_val, 
                                            'y': position.y_val, 
                                            'z': position.z_val}

        # return self.state, reward, False, {}
        return self.state, scale, False, {'x': position.x_val, 
                                           'y': position.y_val, 
                                           'z': position.z_val}
        