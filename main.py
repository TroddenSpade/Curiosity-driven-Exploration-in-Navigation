from matplotlib import pyplot as plt
from envs.DroneEnv import DroneEnv
import ICM
import torch.nn as nn


policy_kwargs = {
    "hidden_layers": (256,),
}

value_kwargs = {
    "hidden_layers": (256,)
}

icm_kwargs = {
    "inverse_hidden_layer": (256,), 
    "forward_hidden_layer": (256, 256),
    "activation_fn": nn.ReLU
}

env = DroneEnv()

agent = ICM.PPO(env, n_steps=256, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                icm_kwargs=icm_kwargs,
                use_fe=True,
                feature_size=256, 
                save_every=8,
                tensorboard_log="./logs/",
                auto_load=True)

agent.train(total_timesteps=20_000)
