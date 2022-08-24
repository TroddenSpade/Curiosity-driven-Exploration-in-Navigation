import os
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
from nn.CNN import FeatureExtractor


class PPO:
    def __init__(self, env, n_steps=256, n_epochs=5, batch_size=64, 
                 lr=1e-3, gamma=0.99, tau=0.95, epsilon=1e-5,
                 vf_weight=0.5, ent_weight=0.01,
                 policy_clip=0.2, value_clip=None, grad_norm=0.5, 
                 policy_kwargs=None, value_kwargs=None,
                 icm_kwargs=None,
                 feature_size=32,
                 use_fe=True,
                 tensorboard_log=None,
                 name="DRONE-PPO-ICM",
                 save_path='./models/PPO/',
                 save_every=None,
                 auto_load=True):
        self.global_step = 0
        self.global_episode = 0
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
        self.save_path = save_path
        self.use_fe = use_fe
        self.feature_size = feature_size
        self.policy_kwargs = policy_kwargs
        self.value_kwargs = value_kwargs
        self.icm_kwargs = icm_kwargs
        self.name = name

        self.save_every = save_every
        self.n_updates = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        self.env = ch.envs.Torch(env)
        self.state = self.env.reset()

        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            action_size = self.env.action_space.n
        else:
            action_size = self.env.action_space.shape[0]
        state_size = self.env.observation_space.shape[0]
        self.params = []
        if use_fe:
            self.feature_extractor = FeatureExtractor(output_size=feature_size).to(self.device)
            self.params += list(self.feature_extractor.parameters())

            self.actor_head = Actor(env, feature_size, **policy_kwargs).to(self.device)
            self.critic_head = Critic(feature_size, **value_kwargs).to(self.device)
        else:
            self.feature_extractor = nn.Identity()
            self.actor_head = Actor(env, state_size, **policy_kwargs).to(self.device)
            self.critic_head = Critic(state_size, **value_kwargs).to(self.device)

        self.icm = ICM(self.is_discrete, 
                       state_size=state_size,
                       action_size=action_size,
                       feature_size=feature_size, **icm_kwargs).to(self.device)

        self.params += list(self.actor_head.parameters()) + \
                        list(self.critic_head.parameters()) + \
                        list(self.icm.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

        if auto_load:
            self.load()
        else:
            if tensorboard_log is not None:
                self.run_name = name + "_" + str(int(time.time()))
                self.writer = SummaryWriter(f"{tensorboard_log}/{self.run_name}")
            else:
                self.writer = None


    def save(self):
        name = "M-" + str(self.global_step)
        path = os.path.join(self.save_path, name)

        os.makedirs(path, exist_ok=True) 

        dic = {}
        dic['global_step'] = self.global_step
        dic['global_episode'] = self.global_episode
        dic['n_steps'] = self.n_steps
        dic['n_epochs'] = self.n_epochs
        dic['batch_size'] = self.batch_size
        dic['lr'] = self.lr
        dic['gamma'] = self.gamma
        dic['tau'] = self.tau
        dic['epsilon'] = self.epsilon
        dic['vf_weight'] = self.vf_weight
        dic['ent_weight'] = self.ent_weight
        dic['policy_clip'] = self.policy_clip
        dic['value_clip'] = self.value_clip
        dic['grad_norm'] = self.grad_norm
        dic['tensorboard_log'] = self.tensorboard_log
        dic['save_path'] = self.save_path
        dic["device"] = self.device
        dic["run_name"] = self.run_name
        dic["feature_size"] = self.feature_size
        dic['use_fe'] = self.use_fe
        torch.save(dic, f"{path}/D.pt")

        if self.use_fe:
            torch.save(self.feature_extractor.state_dict(), path + "/FE.pt")
        torch.save(self.actor_head.state_dict(), path + "/AH.pt")
        torch.save(self.critic_head.state_dict(), path + "/CH.pt")
        torch.save(self.icm.state_dict(), path + "/ICM.pt")

        
    def load(self):
        arr = sorted(os.scandir(self.save_path), key=os.path.getmtime)
        if len(arr) == 0:
            if self.tensorboard_log is not None:
                self.run_name = self.name + "_" + str(int(time.time()))
                self.writer = SummaryWriter(f"{self.tensorboard_log}/{self.run_name}")
            else:
                self.writer = None
            return
        path = arr[-1].path
        print("Loading model from: ", path)

        dic = torch.load(path + "/D.pt")
        self.global_step = dic['global_step']
        self.global_episode = dic['global_episode']
        self.n_steps = dic['n_steps']
        self.n_epochs = dic['n_epochs']
        self.batch_size = dic['batch_size']
        self.lr = dic['lr']
        self.gamma = dic['gamma']
        self.tau = dic['tau']
        self.epsilon = dic['epsilon']
        self.vf_weight = dic['vf_weight']
        self.ent_weight = dic['ent_weight']
        self.policy_clip = dic['policy_clip']
        self.value_clip = dic['value_clip']
        self.grad_norm = dic['grad_norm']
        self.tensorboard_log = dic['tensorboard_log']
        self.save_path = dic['save_path']
        self.device = dic['device']
        self.run_name = dic['run_name']
        self.feature_size = dic['feature_size']
        self.use_fe = dic['use_fe']
        self.params = []

        if self.use_fe:
            self.feature_extractor.load_state_dict(torch.load(path + "/FE.pt"))
            self.feature_extractor.to(self.device)
            self.params += list(self.feature_extractor.parameters())
        self.actor_head.load_state_dict(torch.load(path + "/AH.pt"))
        self.actor_head.to(self.device)
        self.critic_head.load_state_dict(torch.load(path + "/CH.pt"))
        self.critic_head.to(self.device)
        self.icm.load_state_dict(torch.load(path + "/ICM.pt"))
        self.icm.to(self.device)

        self.params += list(self.actor_head.parameters()) + \
                        list(self.critic_head.parameters()) + \
                        list(self.icm.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        if self.tensorboard_log is not None:
            self.writer = SummaryWriter(f"{self.tensorboard_log}/{self.run_name}")
        else:
            self.writer = None


    def get_params(self):
        return self.params


    def policy(self, state):
        state = state.to(self.device)
        x = self.feature_extractor(state)
        return self.actor_head(x)


    def baseline(self, state):
        state = state.to(self.device)
        x = self.feature_extractor(state)
        return self.critic_head(x)


    def network(self, state):
        state = state.to(self.device)
        x = self.feature_extractor(state)
        return self.actor_head(x), self.critic_head(x)

    
    def collect_steps(self, n_steps):
        replay = ch.ExperienceReplay(device=self.device)
        steps = 0

        while steps < n_steps:
            steps += 1
            self.global_step += 1

            with torch.no_grad():
                mass = self.policy(self.state)
            action = mass.sample()

            if self.is_discrete:
                log_prob = mass.log_prob(action)
            else:
                log_prob = mass.log_prob(action).mean(dim=1, keepdim=True)

            next_state, reward, done, _ = self.env.step(action)

            replay.append(self.state,
                        action,
                        reward,
                        next_state,
                        done,
                        log_prob=log_prob)
            
            if done:
                if self.writer is not None:
                    episode_reward, episode_length = self.env.episode()
                    self.writer.add_scalar("episode_length", episode_length, self.global_episode)
                    self.writer.add_scalar("episode_reward", episode_reward, self.global_episode)
                self.state = self.env.reset()
                self.global_episode += 1
            else:
                self.state = next_state

        return replay


    def update(self, replay):
        self.n_updates += 1
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

                masses, new_values = self.network(batch.state())
                
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

                forward_loss, inverse_loss = self.icm(batch.state(), batch.next_state(), actions)
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
            self.writer.add_scalar("forward_loss", np.mean(forward_losses), self.global_step)
            self.writer.add_scalar("inverse_loss", np.mean(inverse_losses), self.global_step)
            self.writer.add_scalar("rewards_mean", rewards.mean().item(), self.global_step)

        if self.save_every and self.n_updates % self.save_every == 0:
            print("Saving...")
            self.save()
        return rewards.sum()


    def train(self, total_timesteps):
        steps = 0

        while steps < total_timesteps:
            steps += self.n_steps
            replay = self.collect_steps(self.n_steps)
            rewards = self.update(replay)

            print(f"{self.global_step} steps - {rewards:.2f} reward")
