import gym
import torch

class TorchEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return torch.from_numpy(obs.__array__()).unsqueeze(0), rew, done, info

    def reset(self):
        return torch.from_numpy(self.env.reset().__array__()).unsqueeze(0)