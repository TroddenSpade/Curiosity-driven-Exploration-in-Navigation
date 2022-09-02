import gym

class CartPole(gym.Env):
    def __init__(self, sparse_reward=True):
        self.env = gym.make("CartPole-v1")
        self.steps = 0
        self.MAX_STEPS = 500
        self.sparse_reward = sparse_reward
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def render(self):
        self.env.render()


    def reset(self):
        self.steps = 0
        self.reward = 0.0
        state = self.env.reset()
        return state


    def step(self, action):
        self.steps += 1
        extrinsic_reward = 0
        state, rew, done, info = self.env.step(action)
        self.reward += rew
        if done:
            if self.steps < self.MAX_STEPS:
                extrinsic_reward = -100
            else:
                extrinsic_reward = 100
        if not self.sparse_reward:
            extrinsic_reward = rew
        return state, extrinsic_reward, done, info


    def episode(self):
        return self.reward, self.steps


if __name__ == '__main__':
    env = CartPole()
    state = env.reset()

    while True:
        action = env.action_space.sample()
        print(action)
        state, reward, done, info = env.step(action)
        env.render()
        print(reward)
        if done:
            state = env.reset()
            print("Done")
            break
        