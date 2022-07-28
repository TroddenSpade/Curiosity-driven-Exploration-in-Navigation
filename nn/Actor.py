import torch
import torch.nn as nn
import cherry as ch
import numpy as np
import gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, env, input_size, hidden_layers=(64, 64), 
                 activation=nn.ReLU, optimizer=torch.optim.Adam, **kwargs):
        super().__init__()
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if is_discrete:
            self.actor_output_size = env.action_space.n
        else:
            self.actor_output_size = env.action_space.shape[0]
        self.input_size = input_size

        layers = []
        last = self.input_size

        for l in hidden_layers:
            layers.append(layer_init(nn.Linear(last, l)))
            layers.append(activation())
            last = l
        layers.append(layer_init(nn.Linear(last, self.actor_output_size), std=0.01))

        self.net = nn.Sequential(*layers)
        self.distribution = ch.distributions.ActionDistribution(env)

        self.optimizer = optimizer(self.parameters(), **kwargs)

    def forward(self, x):
        pi = self.net(x)
        mass = self.distribution(pi)
        return mass