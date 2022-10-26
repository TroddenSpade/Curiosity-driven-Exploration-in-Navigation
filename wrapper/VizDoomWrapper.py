import gym
import vizdoom as vzd

class VizDoomWrapper(gym.Wrapper):
    def __init__(self, env, start_time=10, display_window=False, 
                 game_variables=[vzd.GameVariable.KILLCOUNT]):
        super().__init__(env)
        self.game_variables = game_variables
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['variables'] = {}
        info['variables']["real_rewards"] = self.game.get_total_reward()
        for v in self.game_variables:
            info['variables'][v.name] = self.game.get_game_variable(v)
        killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        if reward >= 1000:
            reward = 1.0
        elif self.game.is_player_dead():
            reward = -1.0
        else:
            reward = 0.0

        if killcount > self.killcount:
            reward += 0.1 * (killcount - self.killcount)
            self.killcount = killcount
        return obs, reward, done, info
    
    def reset(self):
        self.killcount = 0
        obs = self.env.reset()
        return obs