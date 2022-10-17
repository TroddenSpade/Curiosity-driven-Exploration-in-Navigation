import torch.nn as nn
from matplotlib import pyplot as plt

import ICM
from envs.DroneEnvModified import DroneEnvModified


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

env = DroneEnvModified()

agent = ICM.PPO(env, n_steps=1024, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                icm_kwargs=icm_kwargs,
                use_fe=True,
                feature_size=64, 
                save_every=2,
                tensorboard_log="./logs/",
                auto_load=True)

agent.train(total_timesteps=1_000_000)
