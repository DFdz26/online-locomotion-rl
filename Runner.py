import math
import time
import torch

from isaacGymConfig.RobotConfig import RobotConfig
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards
from isaacGymConfig.Curriculum import Curriculum
from isaacgym.torch_utils import *


class Runner:
    curricula: Curriculum

    def __init__(self, policy, learning_algorithm, logger: Logger, config_file, env_config, reward: Rewards,
                 num_actions, terrain_config=None, curricula=None, verbose=False, store_observations=False,
                 history_obj=None, device="cpu"):
        self.history_obj = history_obj
        self.prev_obs = None
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
        self.long_buffer = None
        self.num_terrains = 1 if terrain_config is None else terrain_config.rows
        self.device = device

        self.recording = False

        self.n_steps = 0
        self.n_iterations = 0
        self.starting_training_time = 0
        self.starting_iteration_time = 0
        self.reset_envs_flag = False

        self.logger.set_robot_name(self.agents.get_asset_name())
        self.logger.store_reward_param(self.rewards.reward_terms)
        self.logger.store_curriculum(self.curricula)

        if hasattr(self.learning_algorithm, "get_info_algorithm"):
            self.logger.store_algorithm_parameters(self.learning_algorithm.get_info_algorithm(get_PIBB=False))

        self.obs, self.obs_exp, self.closed_simulation = self.agents.reset_simulation()
        self._store_history()

        if store_observations:
            self.num_observations, self.num_observation_sensor, self.num_expert_observation = self.agents.get_num_observations()

    def _store_history(self):
        if self.history_obj is not None:
            self.prev_obs = self.history_obj.store_history(self.obs)

    def _reset_history(self):
        if self.history_obj is not None:
            self.history_obj.reset_history()

            self.prev_obs = self.history_obj.history

    def _reset_specific_env_history(self, envs):
        if self.history_obj is not None:
            self.history_obj.reset_specific_history(envs)

            self.prev_obs = self.history_obj.history

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
        distance = torch.mean(self.agents.get_final_distance())
        noise = self.learning_algorithm.get_noise()
        best_index = torch.argmax(distance if self.best_distance else rewards)

        if not (noise is None):
            noise = noise.detach().clone()

        self.logger.store_data(distance, rewards, self.learning_algorithm.get_weights_policy(), noise, iteration,
                               total_elapsed_time, self.fallenn_step, self.agents.stored_actions_0leg, 
                               self.agents.stored_desired_0leg, show_plot=False)
        loss_AC, loss_AC_with_supervision = self.learning_algorithm.update(self.policy, rewards)
        self.learning_algorithm.print_info(rewards, iteration, total_elapsed_time, elapsed_time_iteration,
                                           loss_AC, self.long_buffer, loss_AC_with_supervision)
        
        ppo_info = self.learning_algorithm.get_last_PPO_info_for_logger()
        self.logger.store_PPO_run_info(ppo_info, iteration)
        self.logger.plot_learning(iteration)
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
        start_rand = False
        push_robot = False
        new_frequency = None
        frequency_change = None
        change_maximum = False
        minimum = 0.49
        maximum = 0.51
        freq_control = None
        self.starting_training_time = time.time()
        self.learning_algorithm.prepare_training(self.agents, steps_ppo, self.num_observation_sensor,
                                                 self.num_expert_observation, self.num_actions, self.policy)

        for i in range(199):
            self.starting_iteration_time = time.time()

            # if i == 200:
            #     self.agents.get_record_robot()

            # if i == 150:
            #     push_robot = True

            # if i > 100:
            #     freq_control = torch_rand_float(minimum, maximum, (self.agents.get_num_envs(), 1), device=self.device)
            #     freq_control.clamp(min=0.45, max=1.)
            #     maximum += 0.002
            #     minimum -= 0.002
            #     if maximum > 1.0:
            #         maximum = 1.0
            #     if minimum < 0.45:
            #         minimum = 0.45
            #     print("maximum phi:", maximum, "mean: ", freq_control.mean(), "minimum: ", minimum)

            # elif change_maximum:
            #     freq_control = torch.zeros(self.agents.get_num_envs(), 1, dtype=torch.float32, device=self.device, requires_grad=False).fill_(0.5)

            #     # filter_ = torch.less_equal(freq_control, 0.37)

            #     # if len(filter_.nonzero(as_tuple=True)[0]):
            #     #     freq_control[filter_] = 0.

                

                
            for step in range(steps_per_iteration):
                dt = new_frequency

                if step == 75 and push_robot:
                    self.agents.inserte_push()
                    print("pushing")

                if dt is not None:
                    dt = (dt/4)*self.learning_algorithm.get_dt_cpgs()

                actions, ppo_rw, rw_ppo_noise = self.learning_algorithm.act(self.obs, self.obs_exp, self.prev_obs, dt=dt, 
                                                                            frequency_change=None if freq_control is None else torch.flatten(freq_control))

                self.obs, self.obs_exp, actions, reward, \
                    dones, info, closed_simulation = self.agents.step(None, actions,
                                                                      iterations_without_control=new_frequency, freq_control=freq_control)
                reward = self.rewards.include_ppo_reward_penalization(ppo_rw, rw_ppo_noise, reward, self.curricula.get_robot_levels())
                self._store_history()
                self.agents.save_distance()
                # print(step, self.agents.distance.mean())

                if step == (steps_per_iteration - 1):
                    dones.fill_(1)

                if torch.all(dones > 0):
                    self.agents.save_distance()

                self.learning_algorithm.post_step_simulation(self.obs, self.obs_exp, actions, reward, dones, info,
                                                             closed_simulation)
                if self.reset_envs_flag:
                    self.agents.reset_envs(torch.flatten(dones.nonzero()))

                    if not change_maximum:
                        change_maximum = True
                        self.learning_algorithm.change_maximum_frequency_cpg(4.5)

                self._recording_process()
                if closed_simulation or torch.all(dones > 0):
                    break

            if closed_simulation:
                break

            if i == 200:
                self.agents.stop_record_robot()

            print("*******")
            self.fallenn_step = int(torch.count_nonzero(self.agents.env_touching))

            print(f"Fallen robots: {torch.count_nonzero(self.agents.env_touching)}")
            self.long_buffer = torch.mean(self.agents.episode_length_buf.to(torch.float32))
            self._stop_recording()
            final_reward = self.agents.compute_final_reward()
            rewards = final_reward * 0.5
            self.learning_algorithm.last_step(self.obs, self.obs_exp)

            self._learning_process_(i, rewards)

            if (i + 1) != iterations:
                # In case of having curriculum, update it
                if not (self.curricula is None):
                    aux = self.curricula.set_control_parameters(i, final_reward, None,
                                                                self.rewards, self.learning_algorithm,
                                                                steps_per_iteration)
                    steps_per_iteration, update_randomization, started_randomization, self.reset_envs_flag, \
                        new_frequency, body_push = aux

                    if start_rand is False:
                        start_rand = started_randomization

                    if start_rand:
                        self.agents.activate_randomization()
                        self.agents.enable_random_body_vel_beginning()
                    
                    if update_randomization and start_rand:
                        self.agents.get_new_randomization()

                    if body_push:
                        self.agents.enable_random_body_vel_beginning()

                # Reset the environments, the reward buffers and get the first observation

                self.rewards.clean_buffers()
                self.agents.reset_all_envs()
                self._reset_history()
                self.obs, self.obs_exp = self.agents.create_observations()
                self._store_history()

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
