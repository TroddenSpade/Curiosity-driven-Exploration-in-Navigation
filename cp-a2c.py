import gym
from matplotlib import pyplot as plt
import torch.nn as nn

from envs.CartPole import CartPole
from envs.DroneEnv import DroneEnv
import A2C


policy_kwargs = {
    "hidden_layers": (64, 64),
    "activation": nn.Tanh
}

value_kwargs = {
    "hidden_layers": (64, 64),
    "activation": nn.Tanh
}

env = CartPole()

agent = A2C.A2C(env, n_steps=5, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                use_fe=False,
                save_every=None,
                tensorboard_log="./logs/",
                auto_load=False)

agent.train(total_timesteps=100_000)
