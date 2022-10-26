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

from RND.RND import RND
from nn import Actor
from nn import DoubleHeadCritic
from nn import FeatureExtractor


class PPO:
    def __init__(self, env, n_steps=128, n_epochs=4, batch_size=4, 
                 lr=1e-4, ext_gamma=0.999, int_gamma=0.99, tau=0.95, epsilon=1e-5,
                 alpha=1.0, beta=1.0, norm_steps=50, update_proportion=0.25,
                 vf_weight=0.5, ent_weight=0.001,
                 int_coef=1., ext_coef=2.,
                 policy_clip=0.1, value_clip=None, grad_norm=0.5, 
                 policy_kwargs=None, value_kwargs=None,
                 feature_kwargs=None, rnd_kwargs=None,
                 feature_size=512,
                 use_fe=True,
                 tensorboard_log=None,
                 name="PPO-RND",
                 save_path='./models/PPO-RND/',
                 save_every=None,
                 auto_load=True):
        self.global_step = 0
        self.global_rollouts = 0
        self.n_steps = n_steps
        self.norm_steps = norm_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.int_gamma = int_gamma
        self.ext_gamma = ext_gamma
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.update_proportion = update_proportion
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.vf_weight = vf_weight
        self.ent_weight = ent_weight
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.grad_norm = grad_norm
        self.tensorboard_log = tensorboard_log
        self.save_path = save_path
        self.feature_size = feature_size
        self.policy_kwargs = policy_kwargs
        self.value_kwargs = value_kwargs
        self.rnd_kwargs = rnd_kwargs
        self.name = name
        self.use_fe = use_fe

        self.save_every = save_every
        self.n_updates = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.env = env
        self.state = self.env.reset()

        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.cnn = len(self.env.observation_space.shape) > 1
        if self.is_discrete:
            action_size = self.env.action_space.n
            self.action_shape = (1,)
        else:
            action_size = self.env.action_space.shape[0]
            self.action_shape = self.env.action_space.shape
        self.state_shape = self.env.observation_space.shape
        print("State Shape: ", self.state_shape)

        self.params = []
        if use_fe:
            self.feature_extractor = FeatureExtractor(state_shape=self.state_shape, **feature_kwargs).to(self.device)
            self.params += list(self.feature_extractor.parameters())

            self.actor_head = Actor(env, feature_kwargs["feature_size"], **policy_kwargs).to(self.device)
            self.critic_head = DoubleHeadCritic(feature_kwargs["feature_size"], **value_kwargs).to(self.device)
        else:
            self.feature_extractor = nn.Identity()
            self.actor_head = Actor(env, self.state_shape[0], **policy_kwargs).to(self.device)
            self.critic_head = DoubleHeadCritic(self.state_shape[0], **value_kwargs).to(self.device)

        self.rnd = RND(self.is_discrete, 
                       cnn=self.cnn,
                       state_shape=self.state_shape,
                       action_size=action_size,
                       **rnd_kwargs).to(self.device)

        self.params += list(self.actor_head.parameters()) + \
                        list(self.critic_head.parameters()) + \
                        list(self.rnd.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        print("Feature Extractor: ", self.feature_extractor)
        print("Actor: ", self.actor_head)
        print("Critic: ", self.critic_head)

        self.norm_rewards = torch.zeros((self.env.num_envs,)).to(self.device)
        self.int_S1 = torch.zeros((self.env.num_envs,)).to(self.device)
        self.int_S2 = torch.zeros((self.env.num_envs,)).to(self.device)
        self.int_N = 0
        self.state_S1 = torch.zeros(self.state_shape[1:]).to(self.device)
        self.state_S2 = torch.zeros(self.state_shape[1:]).to(self.device)
        self.state_N = 0

        loaded = False
        if auto_load:
            loaded = self.load()

        if not loaded:
            self.normalize_states()
            self.run_name = name + "_" + str(int(time.time()))
        self.mean = self.state_S1
        self.std = torch.sqrt(self.state_S2 - self.state_S1**2)

        if tensorboard_log is not None:
            self.writer = SummaryWriter(f"{tensorboard_log}/{self.run_name}")
        else:
            self.writer = None


    def save(self):
        name = "R-" + str(self.global_step)
        path = os.path.join(self.save_path, name)
        os.makedirs(path, exist_ok=True) 

        dic = {}
        dic['global_step'] = self.global_step
        dic['global_rollouts'] = self.global_rollouts
        dic["run_name"] = self.run_name
        dic['lr'] = self.lr
        torch.save(dic, f"{path}/D.pt")

        norms = {}
        norms["norm_rewards"] = self.norm_rewards
        norms["int_S1"] = self.int_S1
        norms["int_S2"] = self.int_S2
        norms["int_N"] = self.int_N
        norms["state_S1"] = self.state_S1
        norms["state_S2"] = self.state_S2
        norms["state_N"] = self.state_N
        torch.save(norms, f"{path}/N.pt")

        if self.use_fe:
            torch.save(self.feature_extractor.state_dict(), path + "/FE.pt")
        torch.save(self.actor_head.state_dict(), path + "/AH.pt")
        torch.save(self.critic_head.state_dict(), path + "/CH.pt")
        torch.save(self.rnd.state_dict(), path + "/RND.pt")

        
    def load(self):
        arr = sorted(os.scandir(self.save_path), key=os.path.getmtime)
        if len(arr) == 0:
            return False
        path = arr[-1].path
        print("Loading model from: ", path)

        dic = torch.load(path + "/D.pt")
        self.global_step = dic['global_step']
        self.global_rollouts = dic['global_rollouts']
        self.run_name = dic["run_name"]
        self.lr = dic["lr"]

        norms = torch.load(path + "/N.pt")
        self.norm_rewards = norms["norm_rewards"].to(self.device)
        self.int_S1 = norms["int_S1"].to(self.device)
        self.int_S2 = norms["int_S2"].to(self.device)
        self.int_N = norms["int_N"]
        self.state_S1 = norms["state_S1"].to(self.device)
        self.state_S2 = norms["state_S2"].to(self.device)
        self.state_N = norms["state_N"]
        self.params = []

        if self.use_fe:
            self.feature_extractor.load_state_dict(torch.load(path + "/FE.pt"))
            self.feature_extractor.to(self.device)
            self.params += list(self.feature_extractor.parameters())
        self.actor_head.load_state_dict(torch.load(path + "/AH.pt"))
        self.actor_head.to(self.device)
        self.critic_head.load_state_dict(torch.load(path + "/CH.pt"))
        self.critic_head.to(self.device)
        self.rnd.load_state_dict(torch.load(path + "/RND.pt"))
        self.rnd.to(self.device)

        self.params += list(self.actor_head.parameters()) + \
                        list(self.critic_head.parameters()) + \
                        list(self.rnd.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        return True


    def get_params(self):
        return self.params


    def normalize(self, state):
        x = state / 255.
        return ((x - self.mean) / self.std).clip(-5, 5)


    def policy(self, state):
        x = state / 255.
        x = self.feature_extractor(x)
        return self.actor_head(x)


    def baseline(self, state):
        x = state / 255.
        x = self.feature_extractor(x)
        return self.critic_head(x)


    def network(self, state):
        x = state / 255.
        x = self.feature_extractor(x)
        return self.actor_head(x), self.critic_head(x)


    def update_state_norms(self, states):
        x = states / 255.
        total = self.state_N + states.size(0)
        self.state_S1 = self.state_S1 * (self.state_N/total) + x.sum(0)/total
        self.state_S2 = self.state_S2 * (self.state_N/total) + (x**2).sum(0)/total
        self.state_N = total


    def update_returns(self, rewards):
        self.norm_rewards = self.norm_rewards * self.int_gamma + rewards
        return self.norm_rewards


    def update_intrinsic_std(self, returns):
        total = self.int_N + returns.size(0)
        self.int_S1 = self.int_S1 * (self.int_N/total) + returns.sum(0)/total
        self.int_S2 = self.int_S2 * (self.int_N/total) + (returns**2).sum(0)/total
        self.int_N = total
        return torch.sqrt(self.int_S2 - self.int_S1**2)


    def normalize_states(self):
        print('Normalizing states...')
        self.env.reset()
        
        for _ in tqdm(range(self.norm_steps)):
            for step in range(self.n_steps):
                action = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
                state, reward, done, _ = self.env.step(action)

                states = torch.from_numpy(state[:, 3, :, :]).float().to(self.device)
                self.update_state_norms(states)

    
    def collect_steps(self, n_steps):
        replay = ch.ExperienceReplay(device=self.device)
        steps = 0

        while steps < n_steps:
            steps += 1
            self.global_step += 1

            self.state = torch.from_numpy(self.state).float().to(self.device)
            with torch.no_grad():
                mass = self.policy(self.state)
            action = mass.sample()

            if self.is_discrete:
                log_prob = mass.log_prob(action)
            else:
                log_prob = mass.log_prob(action).mean(dim=1)

            next_state, reward, done, info = self.env.step(action.cpu().numpy())

            replay.append(self.state,
                        action,
                        reward,
                        next_state,
                        done,
                        log_prob=log_prob)
            
            self.state = next_state

        self.global_rollouts += 1
        if self.writer is not None:
            dic = {}
            for key in info[0]["variables"].keys():
                val = 0
                for inf in info:
                    val += inf["variables"][key]
                self.writer.add_scalar(f"rollout/{key}", val / self.env.num_envs, self.global_rollouts)
        # print(f"RO:{self.global_rollouts} | {info[:3]} >>>")

        return replay


    def gen_adv(self, rewards, dones, values, next_value, gamma):
        next_values = torch.cat((values[1:], next_value), 0)
        # rewards = rewards.reshape(-1, self.env.num_envs)

        td_errors = rewards + gamma * (1-dones) * next_values - values

        R = torch.zeros_like(td_errors)
        returns = torch.zeros_like(td_errors)
        length = returns.size(0)
        for t in reversed(range(length)):
            R = R * (1.0 - dones[t])
            R = td_errors[t] + self.tau * gamma * R
            returns[t] += R[0]
        
        return returns


    def update(self, replay):
        self.n_updates += 1
        # Logging
        policy_losses = []
        value_losses = []
        entropies = []
        rnd_losses = []

        states = replay.state().reshape(-1, *self.state_shape)
        next_states = replay.next_state().reshape(-1, *self.state_shape)[:, [3], :, :]
        actions = replay.action().reshape(-1, *self.action_shape)
        log_probs = replay.log_prob().flatten()
        dones = replay.done()
        ext_rewards = replay.reward()

        with torch.no_grad():
            next_state_int_value, next_state_ext_value = self.baseline(replay[-1].next_state[0])
            int_values, ext_values = self.baseline(states)
            int_rewards = self.rnd.intrinsic_reward(self.normalize(next_states)).reshape(-1, self.env.num_envs)
        
        int_discounted_rewards = torch.stack(
            [self.update_returns(rews) for rews in int_rewards]
        ).to(self.device)
        std = self.update_intrinsic_std(int_discounted_rewards)
        int_rewards /= std

        # norm states
        self.update_state_norms(next_states)

        int_advantages = self.gen_adv(int_rewards, 
                                      torch.zeros_like(dones), 
                                      int_values.reshape(-1, self.env.num_envs), 
                                      next_state_int_value.reshape(-1, self.env.num_envs), 
                                      self.int_gamma).flatten()
        ext_advantages = self.gen_adv(ext_rewards, 
                                      dones, 
                                      ext_values.reshape(-1, self.env.num_envs), 
                                      next_state_ext_value.reshape(-1, self.env.num_envs), 
                                      self.ext_gamma).flatten()

        int_returns = int_advantages + int_values
        ext_returns = ext_advantages + ext_values
        advantages = int_advantages * self.int_coef + ext_advantages * self.ext_coef

        l = states.size(0)
        n_batches = l // self.batch_size

        # Perform some optimization steps
        for _ in range(self.n_epochs):
            indexes = torch.randperm(l)

            for i in range(n_batches):
                batch_idx = indexes[i*self.batch_size:(i+1)*self.batch_size]

                states_ = states[batch_idx]
                next_states_ = next_states[batch_idx]
                actions_ = actions[batch_idx]
                log_probs_ = log_probs[batch_idx]

                predict_next_state_features, target_next_state_features = self.rnd(self.normalize(next_states_))
                rnd_loss = (predict_next_state_features - target_next_state_features.detach()).pow(2).mean(-1)
                mask = (torch.rand(len(rnd_loss)) < self.update_proportion).float().to(self.device)
                rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))


                masses, (int_new_values, ext_new_values) = self.network(states_)
                if self.is_discrete:
                    new_log_probs = masses.log_prob(actions_.flatten())
                else:
                    new_log_probs = masses.log_prob(actions_).mean(dim=1)

                entropy = masses.entropy().mean()
                advs_ = ch.normalize(advantages[batch_idx], epsilon=1e-8)

                policy_loss = ppo.policy_loss(new_log_probs,
                                            log_probs_,
                                            advs_,
                                            clip=self.policy_clip)

                if self.value_clip is not None:
                    int_value_loss = ppo.state_value_loss(int_new_values,
                                                        int_values[batch_idx],
                                                        int_returns[batch_idx],
                                                        clip=self.value_clip)
                    ext_value_loss = ppo.state_value_loss(ext_new_values,
                                                        ext_values[batch_idx],
                                                        ext_returns[batch_idx],
                                                        clip=self.value_clip)
                else:
                    int_value_loss = a2c.state_value_loss(int_new_values,
                                                        int_values[batch_idx])
                    ext_value_loss = a2c.state_value_loss(ext_new_values,
                                                        ext_values[batch_idx])
                    
                value_loss = 0.5 * (int_value_loss + ext_value_loss)
                ppo_loss = policy_loss - self.ent_weight * entropy + self.vf_weight * value_loss

                loss = self.alpha * ppo_loss + self.beta * rnd_loss

                # Take optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                rnd_losses.append(rnd_loss.item())

        E = ext_rewards.sum().item() / self.env.num_envs
        I = int_rewards.sum().item() / self.env.num_envs
        if self.writer is not None:
            self.writer.add_scalar("ppo/policy_loss", np.mean(policy_losses), self.global_step)
            self.writer.add_scalar("ppo/value_loss", np.mean(value_losses), self.global_step)
            self.writer.add_scalar("ppo/entropy", np.mean(entropies), self.global_step)
            self.writer.add_scalar("rnd/loss", np.mean(rnd_losses), self.global_step)
            self.writer.add_scalar("rnd/ext_rewards_mean", E, self.global_step)
            self.writer.add_scalar("rnd/int_rewards_mean", I, self.global_step)

        if self.save_every and self.n_updates % self.save_every == 0:
            print("Saving...")
            self.save()
        return E, I


    def train(self, total_timesteps):
        steps = 0

        while steps < total_timesteps:
            steps += self.n_steps
            replay = self.collect_steps(self.n_steps)
            rewards = self.update(replay)

            print(f"{self.global_step} steps - {rewards} reward")
