from Environment import DroneEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

N_TIMESTEPS = 10000
env = DroneEnv(env_path='./envs/1.csv')

check_env(env)

model = PPO(CnnPolicy, env, verbose=1, n_steps=512, tensorboard_log="./logs/")

for i in range(20):
    model.learn(total_timesteps=N_TIMESTEPS, 
                eval_env=None, 
                eval_freq=N_TIMESTEPS, 
                n_eval_episodes=5, 
                tb_log_name='PPO', 
                eval_log_path='./logs/',
                reset_num_timesteps=False)
        
    model.save(f'./models/PPO/{(i+1)*N_TIMESTEPS}')