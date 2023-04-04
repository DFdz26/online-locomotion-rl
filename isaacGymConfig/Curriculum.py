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

from isaacgym import terrain_utils
from isaacgym import gymapi
from .TerrainConfig import Terrain
from .TerrainConfig import TerrainComCfg

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


class TerrainCurriculum:
    def __init__(self, num_env, device, cfg : TerrainCurrCfg) -> None:
        if None is cfg.object:
            raise Exception("Object not passed to the Terrain Curriculum Configuration object")
        
        if None is cfg.Control.threshold:
            raise Exception("Threshold not set up for the Terrain Curriculum")
        
        if not (cfg.Control.type in type_curriculums):
            raise Exception(f"Type {self.cfg.Control.type} is not implemented")
        
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
    def __init__(self, num_env, device, terrain_config=None, randomization_config=None) -> None:
        self.terrain_config = terrain_config
        self.randomization_config = randomization_config
        self.num_env = num_env
        self.device = device

        self._set_curriculums_()

    def set_initial_positions(self, positions):
        if self.terrain_curriculm is None:
            return
        
        self.terrain_curriculm.set_initial_position(positions)

    def _set_curriculums_(self):
        self.terrain_curriculm = None if self.terrain_config is None else TerrainCurriculum(self.num_env, self.device, self.terrain_config)
        self.randomization_curriculm = None

    def set_control_parameters(self, iterations, reward):
        if not(self.terrain_curriculm is None):
            self.terrain_curriculm.set_control_parameters(iterations, reward)

        if not(self.randomization_curriculm is None):
            self.randomization_curriculm.set_control_parameters(iterations, reward)

    def get_terrain_curriculum(self, initial_positions):
        if self.terrain_curriculm is None:
            return initial_positions
        
        return self.terrain_curriculm.update(initial_positions)
        

