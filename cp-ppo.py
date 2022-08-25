from matplotlib import pyplot as plt
from envs.CartPole import CartPole
from envs.DroneEnv import DroneEnv
import PPO
import torch.nn as nn


policy_kwargs = {
    "hidden_layers": (32,),
}

value_kwargs = {
    "hidden_layers": (32,)
}

env = CartPole()

agent = PPO.PPO(env, n_steps=256, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                use_fe=False,
                feature_size=32, 
                save_every=None,
                tensorboard_log="./logs/",
                auto_load=False)

agent.train(total_timesteps=100_000)
