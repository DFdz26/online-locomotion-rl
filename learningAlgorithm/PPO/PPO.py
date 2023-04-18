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

from .ActorCritic import ActorCritic
from .Memory import Memory


class PPOArgs:
    # value_loss_coef = 1.1
    # value_loss_coef = 150.
    # value_loss_coef = 300.  # 1500
    # value_loss_coef = 1.  # 1500
    # value_loss_coef = 0.05  # 1500
    value_loss_coef = 0.01  # 1500
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 2  # 5
    # num_mini_batches = 1400  # mini batch size = num_envs*nsteps / nminibatches 400
    num_mini_batches = 1700  # mini batch size = num_envs*nsteps / nminibatches 400
    # num_mini_batches = 400  # mini batch size = num_envs*nsteps / nminibatches
    # learning_rate = 0.0000003  # 5.e-4 and 0.0000003
    # learning_rate = 0.00000012  # 5.e-4 and 0.0000003
    # learning_rate = 0.00000032  # 5.e-4 and 0.000000032
    learning_rate = 1.e-3  # 5.e-4 and 0.000000032
    # learning_rate = 0.000000012  # 5.e-4 and 0.0000003
    # learning_rate = 0.00000015  # 5.e-4
    # schedule = 'fixed'  # could be adaptive, fixed
    schedule = 'adaptive'  # could be adaptive, fixed
    # schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.996
    lam = 0.95
    # desired_kl = 0.01
    desired_kl = 0.011
    max_grad_norm = 1.

    max_clipped_learning_rate = 1.e-2
    # max_clipped_learning_rate = 0.00019
    # # max_clipped_learning_rate = 1.e-2
    # # min_clipped_learning_rate = 5.e-6
    # min_clipped_learning_rate = 0.000008
    min_clipped_learning_rate = 1.e-5

    clipped_values = True
    decay_learning_rate = 0.9992

    multiplyier = 1.0
    decay_multiplyier = 0.95


class PPO:

    def __init__(self, actor_critic: ActorCritic, device='cpu', verbose=False, loading=False):

        self.device = device
        self.verbose = verbose
        self.cfg = PPOArgs()
        self.test = False

        # PPO components
        self.actor_critic = actor_critic
        self.optimizer = None

        if not loading:
            if self.verbose:
                print(f"Copying actorCritic to {device} ...", end='  ')

            self.actor_critic.to(device)

            if self.verbose:
                print(f"Done.")

            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPOArgs.learning_rate)

        self.memory = None  # initialized later
       
        self.step_simulation = Memory.Step()

        self.learning_rate = PPOArgs.learning_rate

    def get_info_algorithm(self, **kwargs):
        return self.cfg

    def prepare_training(self, env_class, steps_per_iteration, num_observations, expert_obs, num_actions, policy):
        self.init_memory(env_class.num_envs, steps_per_iteration, num_observations, expert_obs, num_actions)
        self.train_mode()

    def init_memory(self, num_envs, num_step_simulations_per_env, actor_obs_shape, expert_obs, action_shape):
        self.memory = Memory(num_envs, num_step_simulations_per_env, action_shape,
                             actor_obs_shape, expert_obs, self.device)

    def test_mode(self):
        self.test = True
        # self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def change_kl_distance(self, boost, decay):
        self.cfg.decay_multiplyier = decay
        self.cfg.multiplyier = boost

    def act(self, observation, observation_expert, actions_mult=1.):
        if self.test:
            return self.actor_critic.act_test(observation, observation_expert)
        else:
            self.step_simulation.actions = self.actor_critic.act(observation, observation_expert).detach()
            self.step_simulation.values = self.actor_critic.evaluate(observation, observation_expert).detach()
            self.step_simulation.actions_log_prob = self.actor_critic.get_actions_log_prob(
                self.step_simulation.actions).detach()
            self.step_simulation.action_mean = self.actor_critic.action_mean.detach()
            self.step_simulation.action_sigma = self.actor_critic.action_std.detach()

            self.step_simulation.observations = observation
            self.step_simulation.observation_expert = observation_expert
            self.step_simulation.critic_observations = observation

            # if actions_mult != 1.0:
            #     self.step_simulation.actions *= actions_mult

            return self.step_simulation.actions

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, closed_simulation):
        if info is None:
            info = []

        self.process_env_step(reward, dones, info)

    def process_env_step(self, rewards, dones, infos):
        self.step_simulation.rewards = rewards.clone()
        self.step_simulation.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.step_simulation.rewards += PPOArgs.gamma * torch.squeeze(
                self.step_simulation.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the step_simulation
        self.memory.add_steps_into_memory(self.step_simulation)
        self.step_simulation.clear()
        self.actor_critic.reset()

    def get_policy_weights(self):
        return self.actor_critic.get_weights()

    def load_policy_weights(self, weights):
        self.actor_critic.load_weights(weights)

        if self.verbose:
            print(f"Copying actorCritic to {self.device} ...", end='  ')

        self.actor_critic.to(self.device)

        if self.verbose:
            print(f"Done.")

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPOArgs.learning_rate)

    def last_step(self, last_critic_obs, exp_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, exp_obs).detach()
        self.memory.compute_returns(last_values, PPOArgs.gamma, PPOArgs.lam)

    @staticmethod
    def print_info(rw, rep, total_time, rollout_time, loss):
        mean_fitness = float(torch.mean(rw))

        print("=============================")
        print(f"Rep: {rep}")
        print(f"Mean fitness: {mean_fitness}")
        print(f"Value loss: {loss['mean_value_loss']}", end="\t")
        print(f"Surrogate loss: {loss['mean_surrogate_loss']}")
        print(f"Total time (s): {total_time}")
        print(f"Rollout time (s): {rollout_time}")
        print("=============================")

    def update(self, policy, rewards):
        mean_value_loss = 0
        mean_entropy = 0
        mean_surrogate_loss = 0
        kl_mean_tot = 0
        lr_mean_tot = 0
        mean_loss = 0

        for obs_batch, expert_obs, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, \
            returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch in self.memory.mini_batch_generator(PPOArgs.num_mini_batches, PPOArgs.num_learning_epochs):

            self.actor_critic.act(obs_batch, expert_obs)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, expert_obs)
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
                    kl_mean_tot += float(kl_mean)

                    if kl_mean > (self.cfg.desired_kl * 2.0 * self.cfg.multiplyier):
                        self.learning_rate = max(PPOArgs.min_clipped_learning_rate, self.learning_rate / 1.5)
                    elif ((self.cfg.desired_kl / 2.0) * self.cfg.multiplyier)> kl_mean > 0.0:
                        self.learning_rate = min(PPOArgs.max_clipped_learning_rate, self.learning_rate * 1.5)

                    lr_mean_tot += self.learning_rate

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            elif PPOArgs.schedule == 'decay':
                self.learning_rate *= PPOArgs.decay_learning_rate

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
            if PPOArgs.clipped_values:
                value_clipped = target_values_batch + \
                (value_batch - target_values_batch).clamp(-PPOArgs.clip_param,
                                                            PPOArgs.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPOArgs.value_loss_coef * value_loss - PPOArgs.entropy_coef * entropy_batch.mean()
            # loss = -loss
            mean_loss += float(loss)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPOArgs.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean()

        num_updates = PPOArgs.num_learning_epochs * PPOArgs.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_loss /= num_updates
        kl_mean_tot /= num_updates
        lr_mean_tot /= num_updates
        self.memory.clear()

        if self.cfg.multiplyier < 1.0:
            self.cfg.multiplyier = 1.0
        elif self.cfg.multiplyier > 1.0:
            self.cfg.multiplyier *= self.cfg.decay_multiplyier

        return {"mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "entropy": mean_entropy,
                "lr": lr_mean_tot,
                "mean_loss": mean_loss,
                "kl_mean": kl_mean_tot}

    @staticmethod
    def get_noise():
        return None
