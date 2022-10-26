import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nn.CNN import FeatureExtractor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LinearFE(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(LinearFE, self).__init__()
        self.fc = layer_init(nn.Linear(space_dims, hidden_dims))
        
    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y


class RND(nn.Module):
    def __init__(self, is_discrete=False, 
                 cnn=True,
                 use_fe=True,
                 state_shape=4, action_size=2, 
                 target_hidden_layer=(), 
                 predictor_hidden_layer=(512, 512),
                 feature_hidden_layers=(),
                 feature_size=512,
                 output_size=512,
                 activation_fn=nn.ELU,
                 alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.is_discrete = is_discrete
        self.cnn = cnn

        target_layers = []
        predict_layers = []

        if use_fe:
            if self.cnn:
                target_layers.append(FeatureExtractor((1,) + state_shape[1:], 
                                                      hidden_layers=feature_hidden_layers, 
                                                      feature_size=feature_size, 
                                                      last_activation=bool(len(target_hidden_layer))))
                predict_layers.append(FeatureExtractor((1,) + state_shape[1:], 
                                                       hidden_layers=feature_hidden_layers, 
                                                       feature_size=feature_size))
            else:
                target_layers.append(LinearFE(state_shape[0], feature_size))
                predict_layers.append(LinearFE(state_shape[0], feature_size))

        if is_discrete:
            self.eyes = torch.eye(action_size)

        ## Target Network
        last = feature_size
        for i, l in enumerate(target_hidden_layer):
            target_layers.append(nn.Linear(last, l))
            if i is not len(target_hidden_layer)-1:
                target_layers.append(activation_fn())
            last = l
        self.target_model = nn.Sequential(*target_layers)

        ## Predictor Network
        last = feature_size
        for i, l in enumerate(predictor_hidden_layer):
            predict_layers.append(nn.Linear(last, l))
            if i is not len(predictor_hidden_layer)-1:
                predict_layers.append(activation_fn())
            last = l
        self.predict_model = nn.Sequential(*predict_layers)

        if is_discrete:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = F.mse_loss

        self.target_model.apply(layer_init)
        self.predict_model.apply(layer_init)

        for param in self.target_model.parameters():
            param.requires_grad = False
        
        print("Target Model: ", self.target_model)
        print("Predictor Model: ", self.predict_model)


    def forward(self, state_p):
        target_features = self.target_model(state_p)
        predict_features = self.predict_model(state_p)

        return predict_features, target_features


    def intrinsic_reward(self, state_p):
        target_next_features = self.target_model(state_p)
        predict_next_features = self.predict_model(state_p)
        # r1 = (target_next_features - predict_next_features).pow(2).mean(dim=1, keepdim=True)
        r2 = (target_next_features - predict_next_features).pow(2).sum(1) / 2
        return r2