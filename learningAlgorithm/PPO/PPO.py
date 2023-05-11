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
import torch.nn.functional as F

from .ActorCritic import ActorCritic
from .Memory import Memory


class PPOArgs:
    # value_loss_coef = 1.1
    # value_loss_coef = 150.
    # value_loss_coef = 300.  # 1500
    # value_loss_coef = 1.  # 1500
    # value_loss_coef = 0.05  # 1500
    value_loss_coef = 0.025  # 1500
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 2  # 5
    # num_mini_batches = 1400  # mini batch size = num_envs*nsteps / nminibatches 400
    num_mini_batches = 3000  # mini batch size = num_envs*nsteps / nminibatches 400
    # num_mini_batches = 400  # mini batch size = num_envs*nsteps / nminibatches
    # learning_rate = 0.0000003  # 5.e-4 and 0.0000003
    # learning_rate = 0.00000012  # 5.e-4 and 0.0000003
    # learning_rate = 0.00000032  # 5.e-4 and 0.000000032
    learning_rate = 1.e-3  # 5.e-4 and 0.000000032
    learning_rate_student = 1.e-3  # 5.e-4 and 0.000000032
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
    num_past_actions = 15
    student_module_subsampling = 1

    # coef_differences_CPG = None
    # coef_differences_CPG = 10
    coef_differences_CPG = None

    mini_batches_cpg_def = 100
    learning_rate_cpg_rbfn = 1.e-1


class PPO:

    def __init__(self, actor_critic: ActorCritic, device='cpu', verbose=False, loading=False, cfg=PPOArgs(), store_primitive_movement=False):

        self.device = device
        self.verbose = verbose
        self.cfg = cfg
        self.test = False
        self.expert = False
        self.store_primitive_movement = store_primitive_movement
        self.activated_learning_from_cpg_rbfn = False

        # PPO components
        self.actor_critic = actor_critic
        self.optimizer = None
        self.student_optimizer = None

        if not loading:
            if self.verbose:
                print(f"Copying actorCritic to {device} ...", end='  ')

            self.actor_critic.to(device)

            if self.verbose:
                print(f"Done.")

            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)
            self.student_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate_student)

        self.memory = None  # initialized later

        self.step_simulation = Memory.Step()

        self.learning_rate = self.cfg.learning_rate
        self.optimizer_learn_from_cpg = None

    def get_info_algorithm(self, **kwargs):
        return self.cfg

    def activate_learn_from_cpg_rbfn(self):
        self.activated_learning_from_cpg_rbfn = True
        self.optimizer_learn_from_cpg = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate_cpg_rbfn)

    def deactivate_learn_from_cpg_rbfn(self):
        self.activated_learning_from_cpg_rbfn = False
        self.optimizer_learn_from_cpg = None
        self.memory.clear()

    def prepare_training(self, env_class, steps_per_iteration, num_observations, expert_obs, num_actions, policy):
        self.init_memory(env_class.num_envs, steps_per_iteration, num_observations, expert_obs, num_actions,
                         self.cfg.num_past_actions)
        self.train_mode()

    def init_memory(self, num_envs, num_step_simulations_per_env, actor_obs_shape, expert_obs, action_shape,
                    n_past_observation):
        self.memory = Memory(num_envs, num_step_simulations_per_env, action_shape,
                             actor_obs_shape, expert_obs, n_past_observation * actor_obs_shape, self.device,
                             store_primitive_movement=self.store_primitive_movement)

    def test_mode(self):
        self.test = True
        # self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def change_kl_distance(self, boost, decay):
        self.cfg.decay_multiplyier = decay
        self.cfg.multiplyier = boost

    def change_coef_value(self, new):
        self.cfg.value_loss_coef = new

    def get_encoder_info(self, observation_expert):
        return self.actor_critic.act_expert_encoder(observation_expert)

    def act(self, observation, observation_expert, history, cpg_activations, actions_mult=1.):
        if self.test:
            return self.actor_critic.act_test(observation, observation_expert) if self.expert else \
                self.actor_critic.act_student(observation, history, cpg_activations)
        else:
            self.step_simulation.actions = self.actor_critic.act(observation, observation_expert, cpg_activations).detach()
            self.step_simulation.values = self.actor_critic.evaluate(observation, observation_expert, cpg_activations).detach()
            self.step_simulation.actions_log_prob = self.actor_critic.get_actions_log_prob(
                self.step_simulation.actions).detach()
            self.step_simulation.action_mean = self.actor_critic.action_mean.detach()
            self.step_simulation.action_sigma = self.actor_critic.action_std.detach()

            self.step_simulation.observations = observation
            self.step_simulation.past_observations = history
            self.step_simulation.observation_expert = observation_expert
            self.step_simulation.critic_observations = observation
            self.step_simulation.primitive_movement = cpg_activations.detach()

            # if actions_mult != 1.0:
            #     self.step_simulation.actions *= actions_mult

            return self.step_simulation.actions

    def learn_from_cpg_rbfn(self):
        mean_loss = 0

        for obs_batch, expert_obs, critic_observations_batch, actions_batch, rewards_batch, \
                primitive_movement_batch in self.memory.create_mini_batches_teacher(self.cfg.mini_batches_cpg_def):
            self.actor_critic.act_expert_encoder(expert_obs)
            action = self.actor_critic.act(obs_batch, expert_obs, primitive_movement_batch)
            loss_ = F.mse_loss(action, primitive_movement_batch).mean()

            critic = self.actor_critic.evaluate(critic_observations_batch, expert_obs, primitive_movement_batch)
            loss_ += F.mse_loss(critic, rewards_batch).mean()

            mean_loss += float(loss_)

            self.optimizer.zero_grad()
            loss_.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer_learn_from_cpg.step()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_loss /= num_updates

        self.memory.clear()

        return mean_loss

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, closed_simulation):
        if info is None:
            info = []

        self.process_env_step(reward, dones, info)

    def process_env_step(self, rewards, dones, infos):
        self.step_simulation.rewards = rewards.clone()

        if not self.activated_learning_from_cpg_rbfn:
            self._process_env_step_ppo_learning_(dones, infos)
            # print(self.memory.step)

        self.memory.add_steps_into_memory(self.step_simulation)
        self.step_simulation.clear()
        self.actor_critic.reset()

    def _process_env_step_ppo_learning_(self, dones, infos):
        self.step_simulation.dones = dones

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            self.step_simulation.rewards += self.cfg.gamma * torch.squeeze(
                self.step_simulation.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the step_simulation
        # self.memory.add_steps_into_memory(self.step_simulation)

    def get_policy_weights(self):
        return self.actor_critic.get_weights()

    def load_policy_weights(self, weights):
        self.actor_critic.load_weights(weights)

        if self.verbose:
            print(f"Copying actorCritic to {self.device} ...", end='  ')

        self.actor_critic.to(self.device)

        if self.verbose:
            print(f"Done.")

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)

    def last_step(self, last_critic_obs, exp_obs, primitive_movement_batch):
        last_values = self.actor_critic.evaluate(last_critic_obs, exp_obs, primitive_movement_batch).detach()
        self.memory.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

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

    def save_data_teacher_student_actor(self, observation, observation_expert, cpg_activations):
        self.actor_critic.act_expert_encoder(observation_expert)
        self.step_simulation.actions = self.actor_critic.act(observation, observation_expert, cpg_activations).detach()
        self.step_simulation.observations = observation.detach()
        self.step_simulation.observation_expert = observation_expert.detach()
        self.step_simulation.primitive_movement = cpg_activations.detach()

    def update(self, policy, rewards):
        mean_value_loss = 0
        mean_entropy = 0
        mean_surrogate_loss = 0
        mean_student_loss = 0
        kl_mean_tot = 0
        lr_mean_tot = 0
        mean_loss = 0
        mean_difference_cpg = 0

        for obs_batch, expert_obs, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, \
                returns_batch, old_actions_log_prob_batch, old_mu_batch, \
                old_sigma_batch, past_obs_batch, primitive_movement_batch, rbf_batch in self.memory.mini_batch_generator(self.cfg.num_mini_batches,
                                                                                    self.cfg.num_learning_epochs):
            self.actor_critic.act_expert_encoder(expert_obs)
            new_actions = self.actor_critic.act(obs_batch, expert_obs, primitive_movement_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, expert_obs, primitive_movement_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL for adapting the learning rate.
            # It will observe the distance between the previous and actual policy and try to keep the distance of
            # desired kl, varying the learning rate. Learning rate scheduler.
            if self.cfg.desired_kl is not None and self.cfg.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    kl_mean_tot += float(kl_mean)

                    if kl_mean > (self.cfg.desired_kl * 2.0 * self.cfg.multiplyier):
                        self.learning_rate = max(self.cfg.min_clipped_learning_rate, self.learning_rate / 1.5)
                    elif ((self.cfg.desired_kl / 2.0) * self.cfg.multiplyier) > kl_mean > 0.0:
                        self.learning_rate = min(self.cfg.max_clipped_learning_rate, self.learning_rate * 1.5)

                    lr_mean_tot += self.learning_rate

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            elif self.cfg.schedule == 'decay':
                self.learning_rate *= self.cfg.decay_learning_rate

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg.clip_param,
                                                                               1.0 + self.cfg.clip_param)

            # Surrogates are negatives, so we try to know the maximum of them
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.clipped_values:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.cfg.clip_param,
                                                                          self.cfg.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy_batch.mean()

            difference_cpg_loss = 0.

            if not (self.cfg.coef_differences_CPG is None or primitive_movement_batch is None):
                # difference_cpg_loss = F.mse_loss(primitive_movement_batch, actions_batch) * self.cfg.coef_differences_CPG
                difference_cpg_loss = F.mse_loss(primitive_movement_batch, new_actions) * self.cfg.coef_differences_CPG

            mean_difference_cpg += float(difference_cpg_loss)
            loss += difference_cpg_loss
            # loss_clone = loss.clone()

            mean_loss += float(loss)

            # Gradient step
            self.optimizer.zero_grad()
            # loss_clone.backward(retain_graph=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean()

            for _ in range(self.cfg.student_module_subsampling):
                adaptation_pred = self.actor_critic.student_NN(past_obs_batch)
                with torch.no_grad():
                    adaptation_target = self.actor_critic.expert_NN(expert_obs)

                adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

                self.student_optimizer.zero_grad()
                adaptation_loss.backward()
                self.student_optimizer.step()

                mean_student_loss += adaptation_loss.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_loss /= num_updates
        mean_student_loss /= (num_updates * self.cfg.student_module_subsampling)
        kl_mean_tot /= num_updates
        lr_mean_tot /= num_updates
        mean_difference_cpg /= num_updates
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
                "kl_mean": kl_mean_tot,
                "student_loss": mean_student_loss,
                "mean_difference_cpg": mean_difference_cpg}

    @staticmethod
    def get_noise():
        return None
