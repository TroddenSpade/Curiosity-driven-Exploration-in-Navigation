import torch.nn as nn
from matplotlib import pyplot as plt

import PPO
from envs.DroneEnv import DroneEnv
from envs.DroneEnvModified import DroneEnvModified


policy_kwargs = {
    "hidden_layers": (256,),
}

value_kwargs = {
    "hidden_layers": (256,)
}

env = DroneEnvModified()

agent = PPO.PPO(env, n_steps=1024, 
                policy_kwargs=policy_kwargs, 
                value_kwargs=value_kwargs, 
                use_fe=True,
                feature_size=64, 
                save_every=2,
                tensorboard_log="./logs/",
                name="PPO",
                save_path="./models/PPO",
                auto_load=True)

agent.train(total_timesteps=200_000)
