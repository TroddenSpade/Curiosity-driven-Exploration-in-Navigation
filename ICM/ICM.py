import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.CNN import FeatureExtractor



class DiscreteFE(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(DiscreteFE, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)
        
    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y


class ICM(nn.Module):
    def __init__(self, is_discrete=False, 
                 feature_size=32, 
                 state_size=4, action_size=2, 
                 inverse_hidden_layer=(32,), 
                 forward_hidden_layer=(32,),
                 use_fe=True,
                 activation_fn=nn.ReLU,
                 alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.is_discrete = is_discrete

        if use_fe:
            self.feature_extractor = FeatureExtractor(state_size, feature_size)
        else:
            self.feature_extractor  = nn.Identity()

        if is_discrete:
            self.feature_extractor = DiscreteFE(state_size, feature_size)
            self.eyes = torch.eye(action_size)

        layers = []
        last = feature_size + feature_size
        for l in inverse_hidden_layer:
            layers.append(nn.Linear(last, l))
            layers.append(activation_fn())
            last = l
        layers.append(nn.Linear(last, action_size))
        self.inverse_model = nn.Sequential(*layers)

        layers = []
        last = feature_size + action_size
        for l in forward_hidden_layer:
            layers.append(nn.Linear(last, l))
            layers.append(activation_fn())
            last = l
        layers.append(nn.Linear(last, feature_size))
        self.forward_model = nn.Sequential(*layers)

        if is_discrete:
            self.inverse_loss_fn = nn.CrossEntropyLoss()
        else:
            self.inverse_loss_fn = F.mse_loss


    def forward(self, state, state_p, action):
        x = self.feature_extractor(state)
        x_p = self.feature_extractor(state_p)
        if self.is_discrete:
            action = self.eyes[action.flatten()]
        f_in = torch.cat((x, action), dim=1)
        i_in = torch.cat((x, x_p), dim=1)

        pred_action = self.inverse_model(i_in)
        pred_state_p_feature = self.forward_model(f_in)

        forward_loss = self.forward_loss(pred_state_p_feature, x_p)
        inverse_loss = self.inverse_loss(pred_action, action)

        return forward_loss, inverse_loss


    def forward_pass(self, state, action):
        x = self.feature_extractor(state)
        if self.is_discrete:
            action = self.eyes[action.flatten()]
        x = torch.cat((x, action), dim=1)
        return self.forward_model(x)
        

    def forward_loss(self, pred_state_p, state_p):
        return self.beta * F.mse_loss(pred_state_p, state_p)


    def inverse_loss(self, pred_action, action):
        return (1-self.beta) * self.inverse_loss_fn(pred_action, action)


    def intrinsic_reward(self, state, state_p, action):
        pred_state_p_features = self.forward_pass(state, action)
        state_p_features = self.feature_extractor(state_p)
        return self.alpha * (pred_state_p_features - state_p_features).pow(2).mean(dim=1, keepdim=True)
