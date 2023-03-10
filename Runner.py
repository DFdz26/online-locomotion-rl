import time

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch


from isaacGymConfig.RobotConfig import RobotConfig
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards


class Runner:
    def __init__(self, policy, learning_algorithm, logger: Logger, config_file, env_config, reward: Rewards,
                 num_actions, verbose=False, store_observations=False):
        self.agents = RobotConfig(config_file, env_config, reward, verbose)
        self.policy = policy
        self.logger = logger
        self.learning_algorithm = learning_algorithm
        self.rewards = reward
        self.num_actions = num_actions
        self.num_observations = 0
        self.best_distance = True

        self.n_steps = 0
        self.n_iterations = 0
        self.starting_training_time = 0
        self.starting_iteration_time = 0

        self.logger.set_robot_name(self.agents.get_asset_name())
        self.logger.store_reward_param(self.rewards.reward_terms)

        self.obs, self.closed_simulation = self.agents.reset_simulation()

        if store_observations:


    def _learning_process_(self, iteration, rewards):
        now_time = time.time()

        elapsed_time_iteration = now_time - self.starting_iteration_time
        total_elapsed_time = now_time - self.starting_training_time
        distance = torch.mean(self.agents.compute_env_distance())
        noise = self.learning_algorithm.get_noise()
        best_index = torch.argmax(distance if self.best_distance else rewards)

        if not(noise is None):
            noise = noise.detach().clone()

        self.logger.store_data(distance, rewards, self.policy.get_weights(), noise, iteration, total_elapsed_time,
                               show_plot=True)
        loss = self.learning_algorithm.update(self.policy, rewards)
        self.learning_algorithm.print_info(rewards, iteration, total_elapsed_time, elapsed_time_iteration, loss)

        # Register the next weights and save
        self.logger.store_data_post(self.policy.get_weights())
        self.logger.save_stored_data(actual_weight=self.policy.get_weights(), actual_reward=rewards,
                                     iteration=iteration, total_time=total_elapsed_time, noise=noise,
                                     index=best_index)

    def learn(self, iterations, steps_per_iteration, best_distance=True):
        self.best_distance = best_distance

        closed_simulation = False
        self.starting_training_time = time.time()
        self.learning_algorithm.prepare_training(self.agents, steps_per_iteration, [self.num_observations],
                                                 [self.num_actions], self.policy)

        for i in range(iterations):
            self.starting_iteration_time = time.time()

            for step in range(steps_per_iteration):
                actions = self.learning_algorithm.act(self.obs)

                self.obs, actions, reward, dones, info, closed_simulation = self.agents.step(None, actions)
                self.learning_algorithm.post_step_simulation(self.obs, actions, reward, dones, info, closed_simulation)

                if closed_simulation or torch.all(dones > 0):
                    break

            if closed_simulation:
                break

            rewards = self.agents.compute_final_reward()
            self.learning_algorithm.last_step(self.obs)

            self._learning_process_(i, rewards)

            if (i + 1) != iterations:
                # Reset the environments, the reward buffers and get the first observation
                self.rewards.clean_buffers()
                self.agents.reset_all_envs()
                self.obs = self.agents.create_observations()

    def test_agent(self):
        pass