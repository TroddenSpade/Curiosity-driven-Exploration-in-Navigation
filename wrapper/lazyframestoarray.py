import gym

class LazyFramesToArray(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs.__array__(), rew, done, info

    def reset(self):
        return self.env.reset().__array__()