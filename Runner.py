import math
import time
import torch

from isaacGymConfig.RobotConfig import RobotConfig
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards
from isaacGymConfig.Curriculum import Curriculum


class Runner:
    curricula: Curriculum

    def __init__(self, policy, learning_algorithm, logger: Logger, config_file, env_config, reward: Rewards,
                 num_actions, terrain_config=None, curricula=None, verbose=False, store_observations=False):
        self.agents = RobotConfig(config_file, env_config, reward, terrain_config, curricula, verbose)
        self.policy = policy
        self.logger = logger
        self.learning_algorithm = learning_algorithm
        self.rewards = reward
        self.curricula = curricula
        self.num_actions = num_actions
        self.num_observations = 1
        self.num_observation_sensor = 1
        self.num_expert_observation = 1
        self.best_distance = True
        self.num_terrains = 1 if terrain_config is None else terrain_config.rows

        self.recording = False

        self.n_steps = 0
        self.n_iterations = 0
        self.starting_training_time = 0
        self.starting_iteration_time = 0

        self.logger.set_robot_name(self.agents.get_asset_name())
        self.logger.store_reward_param(self.rewards.reward_terms)
        self.logger.store_curriculum(self.curricula)

        if hasattr(self.learning_algorithm, "get_info_algorithm"):
            self.logger.store_algorithm_parameters(self.learning_algorithm.get_info_algorithm(get_PIBB=False))

        self.obs, self.obs_exp, self.closed_simulation = self.agents.reset_simulation()

        if store_observations:
            self.num_observations, self.num_observation_sensor, self.num_expert_observation = self.agents.get_num_observations()

    def _stop_recording(self):
        if self.recording:
            self.recording = False
            self.agents.stop_recording_videos()

            if self.logger.record_in_progress:
                self.logger.save_videos()

    def _recording_process(self):
        if self.recording:
            self.logger.store_and_save_video(self.agents.recorded_frames)

            if not self.logger.record_in_progress:
                self._stop_recording()

    def _setup_recording(self):
        if self.recording:
            max_difficulty = None if self.curricula is None else self.curricula.get_max_difficulty_terrain()
            terrains = None if max_difficulty is None else [*range(max_difficulty + 1)]
            self.agents.set_up_recording_video(terrains)

    def _learning_process_(self, iteration, rewards):
        now_time = time.time()

        elapsed_time_iteration = now_time - self.starting_iteration_time
        total_elapsed_time = now_time - self.starting_training_time
        distance = torch.mean(self.agents.compute_env_distance())
        noise = self.learning_algorithm.get_noise()
        best_index = torch.argmax(distance if self.best_distance else rewards)

        if not (noise is None):
            noise = noise.detach().clone()

        self.logger.store_data(distance, rewards, self.learning_algorithm.get_weights_policy(), noise, iteration,
                               total_elapsed_time, show_plot=True)
        loss = self.learning_algorithm.update(self.policy, rewards)
        self.learning_algorithm.print_info(rewards, iteration, total_elapsed_time, elapsed_time_iteration, loss)

        # Register the next weights and save
        self.logger.store_data_post(self.learning_algorithm.get_weights_policy())
        self.logger.save_stored_data(actual_weight=self.learning_algorithm.get_weights_policy(), actual_reward=rewards,
                                     iteration=iteration, total_time=total_elapsed_time, noise=noise,
                                     index=best_index)

        if self.logger.record_in_progress:
            self.recording = True
            self._setup_recording()

    def learn(self, iterations, steps_per_iteration, best_distance=True):
        self.best_distance = best_distance
        steps_ppo = steps_per_iteration if self.curricula is None else math.ceil(
            steps_per_iteration / self.curricula.reduce_steps)

        closed_simulation = False
        self.starting_training_time = time.time()
        self.learning_algorithm.prepare_training(self.agents, steps_ppo, self.num_observation_sensor,
                                                 self.num_expert_observation, self.num_actions, self.policy)

        for i in range(iterations):
            self.starting_iteration_time = time.time()

            for step in range(steps_per_iteration):

                actions = self.learning_algorithm.act(self.obs, self.obs_exp)

                self.obs, self.obs_exp, actions, reward, dones, info, closed_simulation = self.agents.step(None,
                                                                                                           actions)
                self.learning_algorithm.post_step_simulation(self.obs, self.obs_exp, actions, reward * 0.5, dones, info,
                                                             closed_simulation)
                self._recording_process()

                if closed_simulation or torch.all(dones > 0):
                    break

            if closed_simulation:
                break

            self._stop_recording()
            final_reward = self.agents.compute_final_reward()
            rewards = final_reward * 0.5
            self.learning_algorithm.last_step(self.obs, self.obs_exp)

            self._learning_process_(i, rewards)

            if (i + 1) != iterations:
                # In case of having curriculum, update it
                if not (self.curricula is None):
                    steps_per_iteration = self.curricula.set_control_parameters(i, final_reward, None,
                                                                                self.rewards, self.learning_algorithm,
                                                                                steps_per_iteration)

                # Reset the environments, the reward buffers and get the first observation
                self.rewards.clean_buffers()
                self.agents.reset_all_envs()
                self.obs, self.obs_exp = self.agents.create_observations()

    def test_agent(self, iterations, steps_per_iteration):
        closed_simulation = False
        self.starting_training_time = time.time()

        for i in range(3):
            self.starting_iteration_time = time.time()

            for step in range(steps_per_iteration):

                actions = self.learning_algorithm.act(self.obs, self.obs_exp)

                self.obs, self.obs_exp, _, _, dones, _, closed_simulation = self.agents.step(None, actions)

                if closed_simulation or torch.all(dones > 0):
                    break

            print(closed_simulation)
            self.agents.reset_all_envs()

            if closed_simulation:
                break

            if (i + 1) != iterations:
                # Reset the environments
                print("a")
                self.agents.reset_all_envs()
                self.obs, self.obs_exp = self.agents.create_observations()
