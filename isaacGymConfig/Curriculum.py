"""
Class: Curriculum
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 3 April 2023

Curriculum 
"""

import numpy as np
import torch
from typing import NewType
import math
import random

from isaacgym import terrain_utils
from isaacgym import gymapi
from .TerrainConfig import Terrain
from .TerrainConfig import TerrainComCfg
from .Rewards import Rewards

TerrainComCfg_type = NewType('TerrainComCfg_type', TerrainComCfg)
Terrain_type = NewType('Terrain_type', Terrain)

type_curriculums = [
    "iterations",
    "reward"
]


class RandomizationCurrCfg:
    class MotorParameters:
        randomize_kp_activated = False
        percentage_kp_range = [[-10, -1], [1, 20]]
        step_randomization_kp = 1

        randomize_kd_activated = False
        percentage_kd_range = [[-10, -1], [1, 20]]
        step_randomization_kd = 1

        randomize_motor_strength = False
        percentage_motor_strength_range = [[-10, -1], [1, 10]]
        step_randomization_motor = 1

    class ModelParameters:
        randomize_payload = False
        payload_range = [[-1, -1], [1, 3]]
        step_randomization_payload = 1

        randomize_friction = False
        friction_range = [[-1, -1], [1, 3]]
        step_randomization_friction = 1

        randomize_restitution = False
        restitution_range = [[0., 0.], [0., 1.0]]
        step_randomization_restitution = 0.1

    class FrequencyControl:
        randomize_frquency_control = False
        randomization_range = [3, 5]

    class Control:
        randomization_activated = False
        generate_first_randomization = False

        increase_range_step = 0
        start_randomization_iteration = 360

        start_randomization_frequency_iteration = 0
        randomization_frequency_iteration = 4

        randomization_interval_iterations = 4


class RandomizationCurriculum:
    def __init__(self, device, cfg: RandomizationCurrCfg):
        self.device = device
        self.cfg = cfg

        self.updates = 0
        self.iteration = 0
        self.level_rand = 0
        self.randomization_frequency_steps = 0

        self.randomization_activated = cfg.Control.randomization_activated
        self.starting_iteration = cfg.Control.start_randomization_iteration
        self.randomization_interval = cfg.Control.randomization_interval_iterations

        self.started = False
        self.range_changed = True
        self.first_iteration = self.cfg.Control.generate_first_randomization

        self._set_up_ranges()
        self.scales_shift = {}
        self.new_freq_control = 0

    @staticmethod
    def _get_scale_shift_ind(_range):
        _scale = 2. / (_range[1] - _range[0])
        shift = (_range[1] + _range[0]) / 2.
        return _scale, shift

    @staticmethod
    def _get_total_range(_total_range, default_value=1., percentage=False):
        _min = _total_range[0][0]
        _max = _total_range[1][1]

        if percentage:
            _min = default_value * (1 + _min/100)
            _max = default_value * (1 + _max/100)

        return [_min, _max]
    
    def get_level(self):
        return self.level_rand

    def get_scales_shift(self, default_motor_strength=1., default_kp=None, default_kd=None):
        if not self.range_changed:
            return self.scales_shift

        scales_shift = {
            "friction": [0., 0.],
            "restitutions": [0., 0.],
            "payloads": [0., 0.],
            "motor_strengths": [0., 0.],
            "kp": [0., 0.],
            "kd": [0., 0.],
        }

        motor_param = self.cfg.MotorParameters
        model_param = self.cfg.ModelParameters

        if len(self.friction_range):
            total_range = self._get_total_range(model_param.friction_range)
            scales_shift["friction"] = self._get_scale_shift_ind(total_range)

        if len(self.restitution_range):
            total_range = self._get_total_range(model_param.restitution_range)
            scales_shift["restitutions"] = self._get_scale_shift_ind(total_range)

        if len(self.payload_range):
            total_range = self._get_total_range(model_param.payload_range)
            scales_shift["payloads"] = self._get_scale_shift_ind(total_range)

        if len(self.motor_strength_range):
            total_range = self._get_total_range(motor_param.percentage_motor_strength_range,
                                                default_motor_strength, percentage=True)
            scales_shift["motor_strengths"] = self._get_scale_shift_ind(total_range)

        # if not(self.kd_range is []):
        #     # TODO: Default kd is actually an array or tensor of max size 3
        #     total_range = self._get_total_range(motor_param.percentage_kd_range * default_kd)
        #     scales_shift["kd"] = self._get_scale_shift_ind(total_range)

        # if not(self.kp_range is []):
        #     # TODO: Default kp is actually an array or tensor of max size 3
        #     total_range = self._get_total_range(motor_param.percentage_kp_range * default_kp)
        #     scales_shift["kp"] = self._get_scale_shift_ind(total_range)

        self.range_changed = False
        self.scales_shift = scales_shift

        return self.scales_shift

    def _check_number_in_range(self, _current_number, _limits, _step, add):
        in_limits = True
        _current_number += int(add) * _step - int(not add) * _step

        if _current_number < _limits[0]:
            _current_number = _limits[0]
            in_limits = False

        elif _current_number > _limits[1]:
            _current_number = _limits[1]
            in_limits = False

        self.range_changed = self.range_changed or in_limits

        return _current_number

    def _increase_range(self, _current_range, _settings_limits, _step):
        if math.ceil(_step) == 0 or _current_range is []:
            return _current_range

        for i in range(len(_current_range)):
            _current_range[i] = self._check_number_in_range(_current_range[i], _settings_limits[i], _step, i == 1)

        return _current_range

    @staticmethod
    def _set_up_static_or_dynamic_range(_range, _step):

        if math.ceil(_step) == 0:
            result = [
                _range[0][0],
                _range[1][1]
            ]
        else:
            result = [
                _range[0][1],
                _range[1][0],
            ]

        return result

    def _set_up_ranges(self):
        model_parameters = self.cfg.ModelParameters
        self.friction_range = [] if not model_parameters.randomize_friction else \
            self._set_up_static_or_dynamic_range(model_parameters.friction_range,
                                                 model_parameters.step_randomization_friction)

        self.payload_range = [] if not model_parameters.randomize_payload else \
            self._set_up_static_or_dynamic_range(model_parameters.payload_range,
                                                 model_parameters.step_randomization_payload)

        self.restitution_range = [] if not model_parameters.randomize_restitution else \
            self._set_up_static_or_dynamic_range(model_parameters.restitution_range,
                                                 model_parameters.step_randomization_restitution)

        motor_parameters = self.cfg.MotorParameters

        self.motor_strength_range = [] if not motor_parameters.randomize_motor_strength else \
            self._set_up_static_or_dynamic_range(motor_parameters.percentage_motor_strength_range,
                                                 motor_parameters.step_randomization_motor)

        self.kd_range = [] if not motor_parameters.randomize_kd_activated else \
            self._set_up_static_or_dynamic_range(motor_parameters.percentage_kd_range,
                                                 motor_parameters.step_randomization_kd)

        self.kp_range = [] if not motor_parameters.randomize_kp_activated else \
            self._set_up_static_or_dynamic_range(motor_parameters.percentage_kp_range,
                                                 motor_parameters.step_randomization_kp)

    @staticmethod
    def _generate_random_tensor(_range, _len, _device, unsqueeze=False):
        _max, _min = _range
        rand_numbers = torch.rand(_len, dtype=torch.float, device=_device, requires_grad=False)

        if unsqueeze:
            rand_numbers = rand_numbers.unsqueeze(1)

        return rand_numbers * (_max - _min) + _min

    def get_model_parameters_randomized(self, num_envs, generate_mass=True):
        friction_ = None
        restitution_ = None
        mass_ = None

        if not self.started and not self.first_iteration:
            return friction_, restitution_, mass_

        self.first_iteration = False

        if not (self.friction_range is []):
            friction_ = self._generate_random_tensor(self.friction_range, num_envs, self.device)

        if not (self.payload_range is []) and generate_mass:
            mass_ = self._generate_random_tensor(self.payload_range, num_envs, self.device)

        if not (self.restitution_range is []):
            restitution_ = self._generate_random_tensor(self.restitution_range, num_envs, self.device)

        return friction_, restitution_, mass_

    def get_motor_parameters_randomized(self, num_envs):
        kp_ = None
        kd_ = None
        motor_strengths_ = None

        if not self.started:
            return kp_, kd_, motor_strengths_

        # if not (self.kp_range is []):
        #     kp_ = self._generate_random_tensor(self.kp_range, num_envs, self.device, unsqueeze=True)

        # if not (self.kd_range is []):
        #     kd_ = self._generate_random_tensor(self.kd_range, num_envs, self.device, unsqueeze=True)

        if not (self.motor_strength_range is []):
            motor_strengths_ = self._generate_random_tensor(self.motor_strength_range, num_envs, self.device,
                                                            unsqueeze=True)
        return kp_, kd_, motor_strengths_

    def _increase_range_process(self):
        model_parameters = self.cfg.ModelParameters
        motor_parameters = self.cfg.MotorParameters

        self.motor_strength_range = self._increase_range(self.motor_strength_range,
                                                         motor_parameters.percentage_motor_strength_range,
                                                         motor_parameters.step_randomization_motor)

        self.kd_range = self._increase_range(self.kd_range,
                                             motor_parameters.percentage_kd_range,
                                             motor_parameters.step_randomization_kd)

        self.kp_range = self._increase_range(self.kp_range,
                                             motor_parameters.percentage_kp_range,
                                             motor_parameters.step_randomization_kp)

        self.payload_range = self._increase_range(self.payload_range,
                                                  model_parameters.payload_range,
                                                  model_parameters.step_randomization_payload)

        self.restitution_range = self._increase_range(self.restitution_range,
                                                      model_parameters.restitution_range,
                                                      model_parameters.step_randomization_restitution)

        self.friction_range = self._increase_range(self.friction_range,
                                                   model_parameters.friction_range,
                                                   model_parameters.step_randomization_friction)
        self.level_rand += 1

    def _vary_frequency(self, iterations):
        if self.cfg.Control.start_randomization_frequency_iteration > iterations:
            return None

        min_rand = self.cfg.FrequencyControl.randomization_range[0]
        max_rand = self.cfg.FrequencyControl.randomization_range[1]

        if self.randomization_frequency_steps == 0:
            self.new_freq_control = random.randint(min_rand, max_rand)

        self.randomization_frequency_steps += 1
        self.randomization_frequency_steps %= self.cfg.Control.randomization_frequency_iteration

        return self.new_freq_control

    def set_control_parameters(self, iterations):
        new_frequency = None

        if self.cfg.FrequencyControl.randomize_frquency_control:
            new_frequency = self._vary_frequency(iterations)

        if not self.randomization_activated:
            return False, False, new_frequency

        self.iteration = iterations
        activated = False

        if self.started:
            self.updates += 1

        if self.started and self.cfg.Control.increase_range_step:
            if self.updates % self.cfg.Control.increase_range_step:
                self._increase_range_process()

        if not self.started and self.starting_iteration == iterations:
            self.started = True
            activated = True
            self.level_rand = 1

        if self.started and self.updates % self.cfg.Control.randomization_interval_iterations == 0:
            return True, activated, new_frequency

        return False, False, new_frequency


class TerrainCurrCfg:
    class Control:
        threshold = None
        step = 0
        type = "iterations"

    percentage_step = 0.7
    object = None


class AlgorithmCurrCfg:
    class PPOCfg:
        gamma = 0.1
        maximum_gamma = 0.5
        step_increase_gamma = 0.1
        iterations_to_increase = 2000
        start_iteration_learning = -999
        divider_initial_steps = 1.5

        start_when_PIBB_stops = True
        activate_increase = False
        stop_learning = False
        scale_rw = True
        change_RW_scales = True
        start_learning_from_CPG_RBFN = False

        boost_kl_distance = 50.
        decay_boost_kl_distance = 0.92
        n_iterations_learning_from_CPG_RBFN = 999

    class PIBBCfg:
        stop_learning = True
        delete_noise_at_stop = True
        control_stop_learning = "iterations"
        threshold = 350

        switching_indirect_to_direct = False
        change_RW_scales_when_switching = True
        control_switching_direct = "iterations"
        threshold_switching = 50

        variance_at_switching = None
        decay_at_switching = None
        boost_first_switching_noise = 1.

        start_at_begining = True
        # decay_influence = 0.996
        decay_influence = 0.996
        iteration_change_rw_indirect = 10

    class StoredDistance:
        activate_store_distance = False
        size = 10


class AlgorithmCurriculum:
    def __init__(self, device, cfg: AlgorithmCurrCfg, verbose=True):
        self.cfg = cfg
        self.device = device
        self.stored_distance = None
        self.gamma = 0.
        self.iteration = 0.
        self.count_increase_gamma = 0
        self.PPO = None
        self.verbose = verbose
        self.start_PPO_with_importance = 0.10

        self.CPG_influence = 1.
        self.start_decrease_CPG = False

        self.PPO_learning_activated = False
        self.PIBB_learning_activated = False
        self.mult_PPO_rw = 1000

        self.PPO_activated = False
        self.PIBB_activated = False
        self.learning_actor_from_cpg = False
        self.iteration_starting_CPG_teacher = self.cfg.PIBBCfg.threshold - \
                                              self.cfg.PPOCfg.n_iterations_learning_from_CPG_RBFN

        if cfg.StoredDistance.activate_store_distance:
            self.stored_distance = torch.zeros(cfg.StoredDistance.size,
                                               dtype=torch.int8,
                                               device=device,
                                               requires_grad=False)

        self._set_up_activated_learning_output_()
        self.switching_CPG_RBFN = self.cfg.PIBBCfg.switching_indirect_to_direct

        if self.verbose:
            if self.cfg.PIBBCfg.decay_influence < 1.:
                print(f"Influence of the CPG will decrease since {self.cfg.PIBBCfg.threshold}")
                print(f"Peace: {self.cfg.PIBBCfg.decay_influence}")

    def change_lr(self):
        self.PPO.change_coef_value(0.025/250)

    def get_NN_weights(self):
        return [self.gamma, self.CPG_influence]

    def _set_up_activated_learning_output_(self):
        if not self.cfg.PPOCfg.start_when_PIBB_stops:
            self.PPO_activated = False
            self.PPO_learning_activated = False

        if self.cfg.PIBBCfg.start_at_begining:
            self.PIBB_activated = True
            self.PIBB_learning_activated = True

    @staticmethod
    def _change_RW_scales_(RewardObj: Rewards):
        rw_weights = RewardObj.get_rewards()

        rw_weights["roll_pitch"]["weight"] *= 2.25
        rw_weights["yaw_vel"]["weight"] *= 1.5 * 2
        # rw_weights["x_velocity"]["weight"] /= 1.1
        rw_weights["x_velocity"]["weight"] /= 1.1
        rw_weights["height_error"]["weight"] /= 10.
        # rw_weights["smoothness"]["weight"] *= 2.
        rw_weights["high_penalization_contacts"]["weight"] *= 7 * 15 * 10 * 5
        rw_weights["velocity_smoothness"]["weight"] *= 2. * 0.00001 * 50
        rw_weights["velocity_smoothness"]["reward_data"]["weight_acc"] *= 1210./500.  # 1200
        rw_weights["slippery"]["weight"] = 1.
        rw_weights["height_error"]["weight"] /= (12 * 3.2)
        # rw_weights["changed_actions"]["weight"] *= 2.5
        #rw_weights["height_error"]["weight"] /= 2

        RewardObj.change_rewards(rw_weights)

    def _change_indirect_to_direct(self, learningObj, rewardObj):
        # Make sure it only happens once
        self.switching_CPG_RBFN = False

        if self.cfg.PIBBCfg.change_RW_scales_when_switching:
            rw_weights = rewardObj.get_rewards()

            rw_weights["yaw_vel"]["weight"] *= 3.2 * 2 * 2 * 3 * 10
            rw_weights["roll_pitch"]["weight"] *= 1.5 * 5
            rw_weights["x_velocity"]["weight"] /= (0.05 * 2.5)
            rw_weights["y_velocity"]["weight"] *= 3
            rw_weights["velocity_smoothness"]["weight"] *= 1.2 * 1.5
            rw_weights["velocity_smoothness"]["reward_data"]["weight_acc"] *= -500. 
            rw_weights["slippery"]["weight"] = 1.1
            rw_weights["high_penalization_contacts"]["weight"] *= 0.0015
            # rw_weights["z_vel"]["weight"] *= 3

            rewardObj.change_rewards(rw_weights)

        # Modify the rbfn - motor neuron connections and copying the learnt weights so far
        CPG_RBFN = learningObj.PIBB.policy
        CPG_RBFN.change_indirect_encoding_to_direct()

        # Get the required info for creating the new noise tensor and cost weighted noise tensor
        len_pibb_noise = CPG_RBFN.get_len_PIBB_noise()
        boost_noise = self.cfg.PIBBCfg.boost_first_switching_noise
        new_variance = self.cfg.PIBBCfg.variance_at_switching
        new_decay = self.cfg.PIBBCfg.decay_at_switching

        # Modify the length of the noise and cost weighted noise tensor creating new ones
        learningObj.PIBB.create_noise_cost_weighted_noise(len_pibb_noise,
                                                          boost_noise=boost_noise,
                                                          new_variance=new_variance,
                                                          new_decay=new_decay)
        
    def change_rw_for_indirect(self, rewardObj):

        rw_weights = rewardObj.get_rewards()

        rw_weights["yaw_vel"]["weight"] *= 3.2 
        rw_weights["roll_pitch"]["weight"] *= 6
        rw_weights["x_velocity"]["weight"] *= 2.5
        rw_weights["y_velocity"]["weight"] *= 1.2

        # rw_weights["z_vel"]["weight"] *= 3

        rewardObj.change_rewards(rw_weights)


    def _start_PPO_(self, RewardObj, steps_per_iteration):
        self.PPO_activated = True
        self.PPO_learning_activated = True
        steps_per_iteration = int(math.floor(steps_per_iteration / self.cfg.PPOCfg.divider_initial_steps))
        self.gamma = self.cfg.PPOCfg.gamma
        self.start_decrease_CPG = True

        if self.cfg.PPOCfg.change_RW_scales:
            self._change_RW_scales_(RewardObj)

        return steps_per_iteration

    def set_control_parameters(self, iterations, reward, distance, RewardObj, learningObj, steps_per_iteration):
        self.iteration = iterations

        if 0 < self.cfg.PIBBCfg.iteration_change_rw_indirect == iterations and self.switching_CPG_RBFN:
            self.change_rw_for_indirect(RewardObj)
        
        if self.start_decrease_CPG and self.cfg.PIBBCfg.decay_influence < 1.:
            self.CPG_influence *= self.cfg.PIBBCfg.decay_influence

            if self.CPG_influence < 0.5:
                self.CPG_influence = 0.5
                self.start_decrease_CPG = False

        if self.switching_CPG_RBFN and self.iteration == self.cfg.PIBBCfg.threshold_switching:
            self._change_indirect_to_direct(learningObj, RewardObj)

        if 0 < self.cfg.PPOCfg.start_iteration_learning == iterations:
            steps_per_iteration = self._start_PPO_(RewardObj, steps_per_iteration)
            self.CPG_influence = 1 - self.start_PPO_with_importance
            self.learning_actor_from_cpg = False

        if 0 < self.iteration_starting_CPG_teacher == iterations and not self.learning_actor_from_cpg:
            self.PPO.activate_learn_from_cpg_rbfn()
            self.learning_actor_from_cpg = True

        if self.cfg.PIBBCfg.stop_learning and self.cfg.PIBBCfg.threshold == iterations:
            self.PIBB_learning_activated = False

            if self.cfg.PPOCfg.start_when_PIBB_stops and not self.PPO_activated:
                steps_per_iteration = self._start_PPO_(RewardObj, steps_per_iteration)
                self.CPG_influence = 1 - self.start_PPO_with_importance
                self.learning_actor_from_cpg = False
                self.PPO.deactivate_learn_from_cpg_rbfn()

            if self.cfg.PIBBCfg.delete_noise_at_stop:
                print("Noise cleared")
                learningObj.PIBB.noise_arr.fill_(0)
                learningObj.PIBB.policy.mn.apply_noise_tensor(learningObj.PIBB.noise_arr)

        if self.PPO_activated and self.cfg.PPOCfg.activate_increase:
            if self.gamma < self.cfg.PPOCfg.maximum_gamma and (
                    self.count_increase_gamma <= self.cfg.PPOCfg.iterations_to_increase):
                self.gamma += self.cfg.PPOCfg.step_increase_gamma
                self.count_increase_gamma = 0
            else:
                self.count_increase_gamma += 1

        return steps_per_iteration, self.PPO_learning_activated

    def _get_error_ppo_cpg_actions(self, PPO_act, PIBB_act):
        error_ppo = torch.zeros()
        if self.PPO_activated:
            error_ppo = torch.norm(PPO_act - PIBB_act)
            return error_ppo

    def get_curriculum_action(self, PPO, PIBB, observations, expert_obs, previous_obs, change_frequency=1.0):
        actions = None
        actions_CPG = None
        amplitude = 1.
        rw_ppo_diff_cpg = None

        if self.PPO_activated:
            encoder_info, amplitude = PPO.get_encoder_info(expert_obs)

        if self.PIBB_activated:
            actions_CPG = PIBB.act(observations, expert_obs, action_mult=2.0, phase_shift=change_frequency) * amplitude
            # rbfn, rbfn_delayed = PIBB.get_rbf_activations()

            actions = actions_CPG

        if self.PPO_activated:

            actions_PPO = PPO.act(observations, expert_obs, previous_obs, actions_CPG, actions_mult=1.)
            rw_ppo_diff_cpg = torch.sum(torch.abs(actions_CPG - actions_PPO), dim=-1)

            # Scale the output to be [-2, 2]
            # for i in range(len(actions_PPO)):
            #     new = [(actions_PPO[i] - torch.min(actions_PPO[i])) / (
            #             torch.max(actions_PPO[i]) - torch.min(actions_PPO[i])) - 0.5] * 4
            #     actions_PPO[i] = new[0]

            if actions is None:
                actions = actions_PPO * self.gamma
            else:
                actions = self.CPG_influence * actions + (1 - self.CPG_influence) * actions_PPO

        elif self.learning_actor_from_cpg:
            PPO.save_data_teacher_student_actor(observations, expert_obs, actions_CPG)

        self.gamma = (1 - self.CPG_influence)

        return actions, rw_ppo_diff_cpg

    @staticmethod
    def change_maximum_change_cpg(PIBB, maximum_freq, dt=None):
        PIBB.change_max_frequency_cpg(maximum_freq, dt)

    def update_curriculum_learning(self, policy, rewards, PPO, PIBB):
        information_Actor_critic = None
        cpg_as_teacher_information = None
        scale_PIBB = 1.
        self.PPO = PPO

        if self.PPO_learning_activated:
            rewards_PPO = rewards * self.mult_PPO_rw  # 1000000

            if self.cfg.PPOCfg.scale_rw:
                rewards_PPO *= self.gamma
                scale_PIBB -= self.gamma

            information_Actor_critic = PPO.update(policy.get_MLP(), rewards_PPO)
        elif self.learning_actor_from_cpg:
            cpg_as_teacher_information = PPO.learn_from_cpg_rbfn()

        if self.PIBB_learning_activated:
            reward_PIBB = scale_PIBB * rewards

            PIBB.update(policy.get_CPG_RBFN(), reward_PIBB)

        return information_Actor_critic, cpg_as_teacher_information

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, PPO, PIBB):
        if self.PIBB_learning_activated:
            PIBB.post_step_simulation(obs, exp_obs, actions, reward, dones, info, False)

        if self.PPO_learning_activated:
            reward_PPO = reward * self.mult_PPO_rw

            if self.cfg.PPOCfg.scale_rw:
                reward_PPO *= self.gamma

            PPO.post_step_simulation(obs, exp_obs, actions, reward_PPO, dones, info, False)

    def last_step_learning(self, obs, exp_obs, PIBB, PPO):
        cpg_mov = PIBB.last_step(obs, exp_obs, reset=(not self.PPO_learning_activated))

        if self.PPO_learning_activated:
            PPO.last_step(obs, exp_obs, cpg_mov)


class TerrainCurriculum:
    def __init__(self, num_env, device, cfg=TerrainCurrCfg()) -> None:
        if None is cfg.object:
            raise Exception("Object not passed to the Terrain Curriculum Configuration object")

        if None is cfg.Control.threshold:
            raise Exception("Threshold not set up for the Terrain Curriculum")

        if not (cfg.Control.type in type_curriculums):
            raise Exception(f"Type {cfg.Control.type} is not implemented")

        self.cfg = cfg
        self.object = Terrain_type(self.cfg.object)
        self.type = self.cfg.Control.type
        self.threshold = self.cfg.Control.threshold
        self.control_step = self.cfg.Control.step
        self.steps = len(self.object.terrain_list) - 1
        self.width_terrains = self.object.config.terrain_width

        self.control_env = torch.zeros(num_env, dtype=torch.int8, device=device, requires_grad=False)
        self.device = device
        self.num_env = num_env

        self.iteration = 0
        self.reward = -999
        self.n_jumps = 0
        self.max_difficulty = 0

    def get_robot_in_level_zero(self):
        return (self.control_env == 0).nonzero()[0][0]

    def get_level(self):
        return self.control_step

    def set_initial_position(self, initial_position):
        self.initial_position = initial_position.detach().clone()

    def set_control_parameters(self, iteration, reward):
        self.iteration = iteration
        self.reward = reward

    def _update_iteration_jump_(self):
        jump = torch.rand(self.num_env, device=self.device) > (1 - self.cfg.percentage_step)
        self.control_env += jump

        limit = torch.nonzero(self.control_env > self.steps).flatten()
        self.n_jumps += 1

        if len(limit):
            self.control_env[limit] = self.steps

        self.max_difficulty = int(torch.max(self.control_env))

    def jump_env_to_terrain(self, env, terrain, initial_position):
        terrain = terrain if terrain < self.steps else self.steps
        initial_position[0] = self.initial_position[env, 0] + self.width_terrains * terrain
        max_z = self.object.get_individual_max_z(initial_position)
        initial_position[2] = self.initial_position[env, 2] + max_z

        return initial_position

    def _change_initial_position(self, initial_position):
        initial_position[:, 0] = self.initial_position[:, 0] + self.width_terrains * self.control_env
        max_z = self.object.get_max_z_heightfield(initial_position)
        initial_position[:, 2] = max_z + self.initial_position[:, 2]

        return initial_position

    def _update_iterations_(self, initial_position):
        change_lr = False

        if type(self.threshold) is list and self.control_step < len(self.threshold):

            if self.iteration == self.threshold[self.control_step]:
                self._update_iteration_jump_()
                self.control_step += 1
                return self._change_initial_position(initial_position), change_lr

        elif type(self.threshold) is dict:
            keys_ = list(self.threshold.keys())

            if self.control_step >= len(keys_):
                return initial_position

            if self.iteration == keys_[self.control_step]:
                self._update_iteration_jump_()
                self.control_step += 1

                change_lr = self.threshold[self.iteration]

                initial_position[:, 0] = self.initial_position[:, 0] + self.width_terrains * self.control_env

        return initial_position, change_lr

    def _update_reward_(self, initial_poistion):
        raise NotImplementedError("Not implemented Terrain update for reward for now")

    def update(self, initial_position):
        return getattr(self, "_update_" + self.type + "_")(initial_position)


class Curriculum:
    def __init__(self, num_env, device, terrain_config=None, randomization_config=None, algorithm_config=None,
                 rw_object=None) -> None:
        self.terrain_config = terrain_config
        self.randomization_config = randomization_config
        self.algorithm_config = algorithm_config
        self.num_env = num_env
        self.device = device
        self.reduce_steps = 1.
        self.rw_object = rw_object

        self._set_curriculums_()

    def get_weights_NN(self):
        weights = {
            "PPO": 1.,
            "PIBB": 1.
        }

        if not (self.algorithm_curriculum is None):
            weights["PPO"], weights["PIBB"] = self.algorithm_curriculum.get_NN_weights()

        return weights

    def set_initial_positions(self, positions):
        if self.terrain_curriculum is None:
            return

        self.terrain_curriculum.set_initial_position(positions)

    def get_max_difficulty_terrain(self):
        if self.terrain_curriculum is None:
            return None

        return self.terrain_curriculum.max_difficulty

    def _set_curriculums_(self):
        self.terrain_curriculum = None if self.terrain_config is None else TerrainCurriculum(self.num_env,
                                                                                             self.device,
                                                                                             self.terrain_config)
        self.algorithm_curriculum = None if self.algorithm_config is None else AlgorithmCurriculum(self.device,
                                                                                                   self.algorithm_config
                                                                                                   )
        self.randomization_curriculum = None if self.randomization_config is None else \
            RandomizationCurriculum(self.device, self.randomization_config)

        if not (self.algorithm_curriculum is None):
            self.reduce_steps = self.algorithm_curriculum.cfg.PPOCfg.divider_initial_steps

    def update_algorithm(self, policy, rewards, PPO, PIBB):
        if not (self.algorithm_curriculum is None):
            return self.algorithm_curriculum.update_curriculum_learning(policy, rewards, PPO, PIBB)

        return None

    def last_step(self, obs, exp_obs, PPO, PIBB):
        if not (self.algorithm_curriculum is None):
            self.algorithm_curriculum.last_step_learning(obs, exp_obs, PIBB, PPO)

    def act_curriculum(self, observation, expert_obs, prev_obs, PPO, PIBB):
        if not (self.algorithm_curriculum is None):
            return self.algorithm_curriculum.get_curriculum_action(PPO, PIBB, observation, expert_obs, prev_obs)

        return None

    def jump_env_to_terrain(self, env, terrain, initial_positions):
        if self.terrain_curriculum is None:
            return initial_positions

        return self.terrain_curriculum.jump_env_to_terrain(env, terrain, initial_positions)

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, PPO, PIBB):
        if not (self.algorithm_curriculum is None):
            self.algorithm_curriculum.post_step_simulation(obs, exp_obs, actions, reward, dones, info, PPO, PIBB)

    def get_randomized_body_properties(self, num_envs, include_mass=True):
        if not (self.randomization_curriculum is None):
            return self.randomization_curriculum.get_model_parameters_randomized(num_envs, include_mass)

        return None, None, None

    def get_randomized_motor_properties(self, num_envs):
        if not (self.randomization_curriculum is None):
            return self.randomization_curriculum.get_motor_parameters_randomized(num_envs)

        return None, None, None

    def get_scales_shift_randomized_parameters(self, default_motor_strength=1., default_kp=None, default_kd=None):
        if not (self.randomization_curriculum is None):
            return self.randomization_curriculum.get_scales_shift(default_motor_strength,
                                                                  default_kp=None, default_kd=None)

        return None

    def set_control_parameters(self, iterations, reward, distance, RwObj, AlgObj, steps_per_iteration):
        new_steps_per_iteration = steps_per_iteration
        randomize_properties = False
        randomized_activated = False
        active_reset_envs = False
        new_frequency = None

        if not (self.terrain_curriculum is None):
            self.terrain_curriculum.set_control_parameters(iterations, reward)

        if not (self.randomization_curriculum is None):
            aux = self.randomization_curriculum.set_control_parameters(iterations)
            randomize_properties, randomized_activated, new_frequency = aux

        if not (self.algorithm_curriculum is None):
            new_steps_per_iteration, active_reset_envs = self.algorithm_curriculum.set_control_parameters(
                iterations,
                reward,
                distance,
                RwObj, AlgObj,
                new_steps_per_iteration)

        return new_steps_per_iteration, randomize_properties, randomized_activated, active_reset_envs, new_frequency

    def get_terrain_curriculum(self, initial_positions):
        if self.terrain_curriculum is None:
            return initial_positions

        initial_positions, change_lr = self.terrain_curriculum.update(initial_positions)

        if change_lr and not (self.algorithm_curriculum is None):
            self.algorithm_curriculum.change_lr()

        return initial_positions

    def randomization_available(self):
        return not (self.randomization_curriculum is None)
    
    def get_levels_curriculum(self):
        terrain = 0
        randomization = 0

        if not (self.terrain_curriculum is None):
            terrain = self.terrain_curriculum.get_level()

        if not (self.randomization_curriculum is None):
            randomization = self.randomization_curriculum.get_level()

        return terrain, randomization
    
    def get_robot_in_level_zero(self):
        idx = 0

        if not (self.terrain_curriculum is None):
            idx = self.terrain_curriculum.get_robot_in_level_zero()

        return idx
