from matplotlib import pyplot as plt
import torch.nn as nn

import ICM
from envs.CartPole import CartPole

env = CartPole()

policy_kwargs = {
    "hidden_layers": (32,),
}

value_kwargs = {
    "hidden_layers": (32,)
}

icm_kwargs = {
    "inverse_hidden_layer": (), 
    "forward_hidden_layer": (),
    "activation_fn": nn.ReLU
}

agent = ICM.PPO(env, n_steps=256, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                icm_kwargs=icm_kwargs,
                use_fe=False,
                name="CARTPOLE-PPO-ICM",
                feature_size=32, tensorboard_log="./logs/")

agent.train(total_timesteps=100_000)