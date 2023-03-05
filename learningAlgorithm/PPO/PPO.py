"""
Class: PPO
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 28 February 2023

PPO learning algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim

from . import ActorCritic
from .Memory import Memory


class PPOArgs:
    value_loss_coef = 1.0
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.


class PPO:
    actor_critic: ActorCritic
    step_simulation: Memory.Step

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.memory = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPOArgs.learning_rate)
        self.step_simulation = Memory.Step()

        self.learning_rate = PPOArgs.learning_rate

    def init_memory(self, num_envs, num_step_simulations_per_env, actor_obs_shape, action_shape):
        self.memory = Memory(num_envs, num_step_simulations_per_env, actor_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, observation):
        self.step_simulation.actions = self.actor_critic.act(observation).detach()
        self.step_simulation.values = self.actor_critic.evaluate(observation).detach()
        self.step_simulation.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.step_simulation.actions).detach()
        self.step_simulation.action_mean = self.actor_critic.action_mean.detach()
        self.step_simulation.action_sigma = self.actor_critic.action_std.detach()

        self.step_simulation.observations = observation
        self.step_simulation.critic_observations = observation
        return self.step_simulation.actions

    def process_env_step(self, rewards, dones, infos):
        self.step_simulation.rewards = rewards.clone()
        self.step_simulation.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.step_simulation.rewards += PPOArgs.gamma * torch.squeeze(
                self.step_simulation.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the step_simulation
        self.memory.add_step_simulations(self.step_simulation)
        self.step_simulation.clear()
        self.actor_critic.reset()

    # TODO: Review
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.memory.compute_returns(last_values, PPOArgs.gamma, PPOArgs.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0

        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, \
            returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch in self.memory.mini_batch_generator(PPOArgs.num_mini_batches, PPOArgs.num_learning_epochs):

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL for adapting the learning rate.
            # It will observe the distance between the previous and actual policy and try to keep the distance of
            # desired kl, varying the learning rate. Learning rate scheduler.
            if PPOArgs.desired_kl is not None and PPOArgs.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPOArgs.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif PPOArgs.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPOArgs.clip_param,
                                                                               1.0 + PPOArgs.clip_param)

            # Surrogates are negatives, so we try to know the maximum of them
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPOArgs.value_loss_coef * value_loss - PPOArgs.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPOArgs.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = PPOArgs.num_learning_epochs * PPOArgs.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.memory.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss
