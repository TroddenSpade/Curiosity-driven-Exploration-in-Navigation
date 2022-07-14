import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, feature_extractor, output_size=2, alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.inverse_model = nn.Sequential(
            nn.Linear(in_features=512 + 512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_size, bias=True),
            nn.Tanh()
        )

        self.forward_model = nn.Sequential(
            nn.Linear(in_features=512 + output_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
        )
        self.feature_extractor = feature_extractor


    def forward_pass(self, state, action):
        x = self.feature_extractor(state)
        x = torch.cat((x, action), dim=1)
        return self.forward_model(x)
        

    def inverse_pass(self, state, state_p):
        x = self.feature_extractor(state)
        x_p = self.feature_extractor(state_p)
        x = torch.cat((x, x_p), dim=1)
        return self.inverse_model(x)

    
    def forward_loss(self, pred_state_p, state_p):
        return self.beta * F.mse_loss(pred_state_p, state_p)


    def inverse_loss(self, pred_action, action):
        return (1-self.beta) * F.mse_loss(pred_action, action)


    def intrinsic_reward(self, state, state_p, action):
        pred_state_p_features = self.forward_pass(state, action)
        state_p_features = self.feature_extractor(state_p)
        return self.alpha * (pred_state_p_features - state_p_features).pow(2).mean(dim=1, keepdim=True)