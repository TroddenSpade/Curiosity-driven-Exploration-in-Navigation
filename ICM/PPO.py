import random
import gym
import time

import numpy as np
import cherry as ch
import cherry.algorithms.ppo as ppo
import cherry.algorithms.a2c as a2c

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ICM.ICM import ICM
from nn.Actor import Actor
from nn.Critic import Critic


class PPO:
    def __init__(self, Env, n_steps=256, n_epochs=5, batch_size=64, 
                 lr=1e-3, gamma=0.99, tau=0.95, epsilon=1e-5,
                 vf_weight=0.5, ent_weight=0.01,
                 policy_clip=0.2, value_clip=None, grad_norm=0.5, 
                 tensorboard_log=None, name="DRONE-PPO-ICM"):
        self.global_step = 0
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.vf_weight = vf_weight
        self.ent_weight = ent_weight
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.grad_norm = grad_norm
        self.tensorboard_log = tensorboard_log

        if tensorboard_log is not None:
            run_name = name + "_" + str(int(time.time()))
            self.writer = SummaryWriter(f"{tensorboard_log}/{run_name}")
        else:
            self.writer = None

        env = Env(sparse_reward=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        self.env = ch.envs.Torch(env)
        self.state = self.env.reset()
        self.state = self.preprocess(self.state)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=25088, out_features=512, bias=True),
            nn.ReLU(),
        )
        self.actor_head = Actor(env, 512, hidden_layers=(), activation=nn.ReLU)
        self.critic_head = Critic(512, hidden_layers=(), activation=nn.ReLU)
        self.icm = ICM(self.feature_extractor)

        self.params = list(self.feature_extractor.parameters()) + \
                 list(self.actor_head.parameters()) + \
                 list(self.critic_head.parameters()) + \
                 list(self.icm.inverse_model.parameters()) + \
                 list(self.icm.forward_model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)


    def get_params(self):
        return self.params

    def preprocess(self, state):
        state = torch.unsqueeze(state, 0)
        state = torch.permute(state, (0, 3, 1, 2))
        return state


    def policy(self, state):
        x = self.feature_extractor(state)
        return self.actor_head(x)


    def baseline(self, state):
        x = self.feature_extractor(state)
        return self.critic_head(x)


    def network(self, state):
        x = self.feature_extractor(state)
        return self.actor_head(x), self.critic_head(x)

    
    def collect_steps(self, n_steps):
        replay = ch.ExperienceReplay()
        steps = 0

        while steps < n_steps:
            steps += 1
            self.global_step += 1

            with torch.no_grad():
                mass = self.policy(self.state)
            action = mass.sample()
            log_prob = mass.log_prob(action).mean(dim=1, keepdim=True)

            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess(next_state)

            replay.append(self.state,
                        action,
                        reward,
                        next_state,
                        done,
                        log_prob=log_prob)
            
            if done:
                self.state = self.env.reset()
                self.state = self.preprocess(self.state)
            else:
                self.state = next_state

        return replay


    def update(self, replay):
        # Logging
        policy_losses = []
        value_losses = []
        entropies = []
        inverse_losses = []
        forward_losses = []

        with torch.no_grad():
            next_state_value = self.baseline(replay[-1].next_state)
            values = self.baseline(replay.state())
            intrinsic_reward = self.icm.intrinsic_reward(replay.state(), replay.next_state(), replay.action())

        rewards = replay.reward() + intrinsic_reward
        advantages = ch.generalized_advantage(self.gamma,
                                                self.tau,
                                                rewards,
                                                replay.done(),
                                                values,
                                                next_state_value)
        returns = advantages + values

        for i, sars in enumerate(replay):
            sars.returns = returns[i]
            sars.advantage = advantages[i]

        l = len(replay._storage)
        n_batches = l // self.batch_size

        # Perform some optimization steps
        for _ in range(self.n_epochs):
            random.shuffle(replay._storage)

            for s_idx in range(n_batches):
                start = self.batch_size * s_idx        
                batch = ch.ExperienceReplay(replay[start: start + self.batch_size])

                state_features = self.feature_extractor(batch.state())
                state_p_features = self.feature_extractor(batch.next_state())

                forward_input = torch.cat((state_features, batch.action()), dim=1)
                pred_state_p_features = self.icm.forward_model(forward_input)
                forward_loss = self.icm.forward_loss(pred_state_p_features, state_p_features)

                inverse_input = torch.cat((state_features, state_p_features), dim=1)
                pred_action = self.icm.inverse_model(inverse_input)
                inverse_loss = self.icm.inverse_loss(pred_action, batch.action())

                masses = self.actor_head(state_features)
                new_values = self.critic_head(state_features)
                
                actions = batch.action()
                new_log_probs = masses.log_prob(actions).mean(dim=1, keepdim=True)
                entropy = masses.entropy().mean()

                advs = ch.normalize(batch.advantage(), epsilon=1e-8)
            
                policy_loss = ppo.policy_loss(new_log_probs,
                                            batch.log_prob(),
                                            advs,
                                            clip=self.policy_clip)
                if self.value_clip is not None:
                    value_loss = ppo.state_value_loss(new_values,
                                                    batch.value(),
                                                    batch.returns(),
                                                    clip=self.value_clip)
                else:
                    value_loss = a2c.state_value_loss(new_values,
                                                    batch.returns())
            
                ppo_loss = policy_loss - self.ent_weight * entropy + self.vf_weight * value_loss

                loss = ppo_loss + forward_loss + inverse_loss
                # Take optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                inverse_losses.append(inverse_loss.item())
                forward_losses.append(forward_loss.item())

        if self.writer is not None:
            self.writer.add_scalar("policy_loss", np.mean(policy_losses), self.global_step)
            self.writer.add_scalar("value_loss", np.mean(value_losses), self.global_step)
            self.writer.add_scalar("entropy", np.mean(entropies), self.global_step)
            self.writer.add_scalar("forward_loss", np.mean(inverse_losses), self.global_step)
            self.writer.add_scalar("inverse_loss", np.mean(forward_losses), self.global_step)

        return rewards.mean()


    def train(self, total_timesteps):
        steps = 0

        while steps < total_timesteps:
            steps += self.n_steps
            replay = self.collect_steps(self.n_steps)
            rewards = self.update(replay)

            print(f"{self.global_step} steps - {rewards:.2f} reward")
