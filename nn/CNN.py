import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

''' Learning by Prediction ICLR 2017 paper
    (their final output was 64 changed to 256 here)
    input: [None, 144, 256, 1]; output: [None, 1344] -> [None, 256];
'''
class FeatureExtractor_(nn.Module):
    def __init__(self, output_size=64):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(in_features=3136, out_features=output_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, state):
        x = F.normalize(state)
        x = self.feature_extractor(state)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, state_shape, hidden_layers=(), feature_size=64, last_activation=True):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        fake_in = torch.rand((1,) + state_shape)
        out = self.cnn(fake_in)

        layers = []
        next = out.shape[1]
        for l in hidden_layers:
            layers.append(nn.Linear(in_features=next, out_features=l, bias=True))
            layers.append(nn.ELU())
            next = l
        layers.append(nn.Linear(in_features=next, out_features=feature_size, bias=True))
        if last_activation:
            layers.append(nn.ELU())

        self.linear = nn.Sequential(*layers)

    def forward(self, state):
        x = self.cnn(state)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    state = torch.randn(1, 4, 84, 84)
    feature_extractor = FeatureExtractor()
    x = feature_extractor(state)
    print(x.shape)