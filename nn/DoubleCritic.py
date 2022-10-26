import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DoubleHeadCritic(nn.Module):
    def __init__(self, input_size, hidden_layers=(64, 64), activation=nn.ELU,
                 optimizer=torch.optim.Adam, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.critic_output_size = 1

        layers = []
        last = self.input_size

        for l in hidden_layers:
            layers.append(layer_init(nn.Linear(last, l)))
            layers.append(activation())
            last = l
        self.net = nn.Sequential(*layers)

        self.value_in = layer_init(nn.Linear(last, self.critic_output_size), std=1.)
        self.value_ex = layer_init(nn.Linear(last, self.critic_output_size), std=1.)

        self.optimizer = optimizer(self.parameters(), **kwargs)

    def forward(self, x):
        value = self.net(x)
        v_in = self.value_in(value + x)
        v_ex = self.value_ex(value + x)
        return v_in, v_ex