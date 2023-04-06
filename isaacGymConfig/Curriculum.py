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

type_controls = [
    "terrain",
    "randomization",
]


class RandmizationCurrCfg:
    name_type = "randomization"

    class MotorParameters:
        percentage_kp_noise = [0, 20]
        step_noise_kp = 1

        percentage_kd_noise = [0, 20]
        step_noise_kd = 1

        percentage_motor_strength = [0, 10]
        step_noise_motor = 1
    
    class ModelParameters:
        percentage_mass_noise = [0, 10]
        percentage_com_noise = [0, 10]

    class Observations:
        percentage_observation_noise = [0, 10]
        step_noise_observation = 1

    class Control:
        threshold = None
        steps = 0.
        type = "iterations"    


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

    class PIBBCfg:
        stop_learning = True
        delete_noise_at_stop = True
        control_stop_learning = "iterations"
        threshold = 350

        start_at_begining = True

    class StoredDistance:
        activate_store_distance = False
        size = 10


class AlgorithmCurriculum:
    def __init__(self, device, cfg: AlgorithmCurrCfg):
        self.cfg = cfg
        self.device = device
        self.stored_distance = None
        self.gamma = 0.
        self.iteration = 0.
        self.count_increase_gamma = 0

        self.PPO_learning_activated = False
        self.PIBB_learning_activated = False
        self.mult_PPO_rw = 1000

        self.PPO_activated = False
        self.PIBB_activated = False

        if cfg.StoredDistance.activate_store_distance:
            self.stored_distance = torch.zeros(cfg.StoredDistance.size,
                                               dtype=torch.int8,
                                               device=device,
                                               requires_grad=False)

        self._set_up_activated_learning_output_()

    def get_NN_weights(self):
        return [self.gamma, 1.]

    def _set_up_activated_learning_output_(self):
        if not self.cfg.PPOCfg.start_when_PIBB_stops:
            self.PPO_activated = False
            self.PPO_learning_activated = False

        if self.cfg.PIBBCfg.start_at_begining:
            self.PIBB_activated = True
            self.PIBB_learning_activated = True

    @staticmethod
    def _change_RW_scales_(RewardObj:Rewards):
        rw_weights = RewardObj.get_rewards()

        rw_weights["roll_pitch"]["weight"] *= 1.5
        rw_weights["x_velocity"]["weight"] *= 1.12

        RewardObj.change_rewards(rw_weights)

    def _start_PPO_(self, RewardObj, steps_per_iteration):
        self.PPO_activated = True
        self.PPO_learning_activated = True
        steps_per_iteration = int(math.floor(steps_per_iteration/self.cfg.PPOCfg.divider_initial_steps))
        self.gamma = self.cfg.PPOCfg.gamma

        if self.cfg.PPOCfg.change_RW_scales:
            self._change_RW_scales_(RewardObj)

        return steps_per_iteration

    def set_control_parameters(self, iterations, reward, distance, RewardObj, learningObj, steps_per_iteration):
        self.iteration = iterations

        if 0 < self.cfg.PPOCfg.start_iteration_learning == iterations:
            steps_per_iteration = self._start_PPO_(RewardObj, steps_per_iteration)

        if self.cfg.PIBBCfg.stop_learning and self.cfg.PIBBCfg.threshold == iterations:
            self.PIBB_learning_activated = False

            if self.cfg.PPOCfg.start_when_PIBB_stops and not self.PPO_activated:
                steps_per_iteration = self._start_PPO_(RewardObj, steps_per_iteration)

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

        return steps_per_iteration

    def get_curriculum_action(self, PPO, PIBB, observations, expert_obs):
        actions = None

        if self.PIBB_activated:
            actions_CPG = PIBB.act(observations, expert_obs) * 1.

            actions = actions_CPG

        if self.PPO_activated:
            actions_PPO = PPO.act(observations, expert_obs, actions_mult=1.)

            # Scale the output to be [-2, 2]
            for i in range(len(actions_PPO)):
                new = [(actions_PPO[i] - torch.min(actions_PPO[i])) / (
                            torch.max(actions_PPO[i]) - torch.min(actions_PPO[i])) - 0.5] * 4
                actions_PPO[i] = new[0]

            if actions is None:
                actions = actions_PPO * self.gamma
            else:
                actions += self.gamma * actions_PPO

        return actions

    def update_curriculum_learning(self, policy, rewards, PPO, PIBB):
        information_Actor_critic = None
        scale_PIBB = 1.

        if self.PPO_learning_activated:
            rewards_PPO = rewards * self.mult_PPO_rw  # 1000000

            if self.cfg.PPOCfg.scale_rw:
                rewards_PPO *= self.gamma
                scale_PIBB -= self.gamma

            information_Actor_critic = PPO.update(policy.get_MLP(), rewards_PPO)

        if self.PIBB_learning_activated:
            reward_PIBB = scale_PIBB * rewards

            PIBB.update(policy.get_CPG_RBFN(), reward_PIBB)

        return information_Actor_critic

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, PPO, PIBB):
        if self.PIBB_learning_activated:
            PIBB.post_step_simulation(obs, exp_obs, actions, reward, dones, info, False)

        if self.PPO_learning_activated:
            reward_PPO = reward * self.mult_PPO_rw

            if self.cfg.PPOCfg.scale_rw:
                reward_PPO *= self.gamma

            PPO.post_step_simulation(obs, exp_obs, actions, reward_PPO, dones, info, False)

    def last_step_learning(self, obs, exp_obs, PIBB, PPO):
        if self.PIBB_learning_activated:
            PIBB.last_step(obs, exp_obs)

        if self.PPO_learning_activated:
            PPO.last_step(obs, exp_obs)

class TerrainCurriculum:
    def __init__(self, num_env, device, cfg: TerrainCurrCfg) -> None:
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

    def set_initial_position(self, initial_position):
        self.initial_position = initial_position.detach().clone()

    def set_control_parameters(self, iteration, reward):
        self.iteration = iteration
        self.reward = reward

    def _update_iteration_jump_(self):
        jump = torch.rand(self.num_env, device=self.device) > (1 - self.cfg.percentage_step)
        self.control_env += jump

        limit = torch.nonzero(self.control_env > self.steps).flatten()

        if len(limit):
            self.control_env[limit] = self.steps

    def _update_iterations_(self, initial_position):
        if type(self.threshold) is list and self.control_step < len(self.threshold):

            if self.iteration == self.threshold[self.control_step]:
                self._update_iteration_jump_()
                self.control_step += 1

                initial_position[:, 0] = self.initial_position[:, 0] + self.width_terrains * self.control_env       

        elif type(self.threshold) is int:
            if self.iteration == self.threshold:
                self._update_iteration_jump_()

                self.threshold += self.control_step

                initial_position[:, 0] = self.initial_position[:, 0] + self.width_terrains * self.control_env
            
        return initial_position

    def _update_reward_(self, initial_poistion):
        raise NotImplementedError("Not implemented Terrain update for reward for now")

    def update(self, initial_position):
        return getattr(self, "_update_" + self.type + "_")(initial_position)
        

class Curriculum:
    def __init__(self, num_env, device, terrain_config=None, randomization_config=None, algorithm_config=None) -> None:
        self.terrain_config = terrain_config
        self.randomization_config = randomization_config
        self.algorithm_config = algorithm_config
        self.num_env = num_env
        self.device = device
        self.reduce_steps = 1.

        self._set_curriculums_()

    def get_weights_NN(self):
        weights = {
            "PPO": 1.,
            "PIBB": 1.
        }

        if not(self.algorithm_curriculum is None):
            weights["PPO"], weights["PIBB"] = self.algorithm_curriculum.get_NN_weights()

        return weights

    def set_initial_positions(self, positions):
        if self.terrain_curriculum is None:
            return
        
        self.terrain_curriculum.set_initial_position(positions)

    def _set_curriculums_(self):
        self.terrain_curriculum = None if self.terrain_config is None else TerrainCurriculum(self.num_env, self.device, self.terrain_config)
        self.algorithm_curriculum = None if self.algorithm_config is None else AlgorithmCurriculum( self.device, self.algorithm_config)
        self.randomization_curriculum = None

        if not(self.algorithm_curriculum is None):
            self.reduce_steps = self.algorithm_curriculum.cfg.PPOCfg.divider_initial_steps

    def update_algorithm(self, policy, rewards, PPO, PIBB):
        if not (self.algorithm_curriculum is None):
            return self.algorithm_curriculum.update_curriculum_learning(policy, rewards, PPO, PIBB)

        return None

    def last_step(self, obs, exp_obs, PPO, PIBB):
        if not(self.algorithm_curriculum is None):
            self.algorithm_curriculum.last_step_learning(obs, exp_obs, PIBB, PPO)

    def act_curriculum(self, observation, expert_obs, PPO, PIBB):
        if not(self.algorithm_curriculum is None):
            return self.algorithm_curriculum.get_curriculum_action(PPO, PIBB, observation, expert_obs)

        return None

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, PPO, PIBB):
        if not(self.algorithm_curriculum is None):
            self.algorithm_curriculum.post_step_simulation(obs, exp_obs, actions, reward, dones, info, PPO, PIBB)

    def set_control_parameters(self, iterations, reward, distance, RwObj, AlgObj, steps_per_iteration):
        new_steps_per_iteration = steps_per_iteration

        if not(self.terrain_curriculum is None):
            self.terrain_curriculum.set_control_parameters(iterations, reward)

        if not(self.randomization_curriculum is None):
            self.randomization_curriculum.set_control_parameters(iterations, reward)

        if not(self.algorithm_curriculum is None):
            new_steps_per_iteration = self.algorithm_curriculum.set_control_parameters(iterations, reward, distance, RwObj, AlgObj, new_steps_per_iteration)

        return new_steps_per_iteration

    def get_terrain_curriculum(self, initial_positions):
        if self.terrain_curriculum is None:
            return initial_positions
        
        return self.terrain_curriculum.update(initial_positions)
        

