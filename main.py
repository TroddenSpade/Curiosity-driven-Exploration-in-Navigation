from Environment import DroneEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


env = DroneEnv()

check_env(env)

model = PPO(CnnPolicy, env, verbose=1, n_steps=256, tensorboard_log="./logs/")

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



print("PPO")
model.learn(total_timesteps=20000)

print("Eval")
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")