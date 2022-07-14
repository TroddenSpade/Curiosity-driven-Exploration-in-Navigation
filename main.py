from envs.DroneEnv import DroneEnv
import numpy as np

import ICM

agent = ICM.PPO(DroneEnv, n_steps=128,  tensorboard_log="./logs/")

agent.train(total_timesteps=10_000)
