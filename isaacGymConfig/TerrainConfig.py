"""
Class: Terrain
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 3 April 2023

Terrain creation
"""

import numpy as np
import math
import torch

from isaacgym import terrain_utils
from isaacgym import gymapi
import time

accepted_terrains = [
    "random_uniform_terrain", 
    "flat_terrain", 
    "sloped_terrain", 
    "pyramid_sloped_terrain", 
    "discrete_obstacles_terrain", 
    "wave_terrain",
    "stairs_terrain",
    "pyramid_stairs_terrain",
    "stepping_stones_terrain",
    ]


class TerrainComCfg:
    terrain_width = 16.
    terrain_length = 12.
    horizontal_scale = 0.25  # [m]
    vertical_scale = 0.005  # [m]
    slope_threshold = 1.5
    columns = 1
    border_x = -2
    border_y = 2


class Terrain:
    def __init__(self, device, num_envs, terrain_list, ComConfig : TerrainComCfg) -> None:
        self.terrain_match = {
            "random_uniform_terrain": self._random_uniform_terrain_,
            "flat_terrain": self._flat_terrain_, 
            "sloped_terrain": self._sloped_terrain_, 
            "pyramid_sloped_terrain": self._pyramid_sloped_terrain_, 
            "discrete_obstacles_terrain": self._discrete_obstacles_terrain_, 
            "wave_terrain": self._wave_terrain_,
            "stairs_terrain": self._stairs_terrain_,
            "pyramid_stairs_terrain": self._pyramid_stairs_terrain_,
            "stepping_stones_terrain": self._stepping_stones_terrain_,
        }
        self.num_envs = num_envs
        self.terrain_list = terrain_list
        self.device = device
        self.config = ComConfig
        self.num_terrains = len(self.terrain_list) * self.config.columns
        self.rows = len(self.terrain_list)
        self.information_terrain = []
        self.observation_terrain = torch.zeros(self.num_envs, len(self.terrain_list), device=self.device, requires_grad=False)
        self._config_terrain_()

    def get_info_terrain(self, positions):
        level = torch.floor((positions[:, 0] - self.config.border_x)/(self.config.terrain_width + 0.1)).to(torch.uint8)
        return torch.FloatTensor([self.information_terrain[int(i)] for i in level]).to(self.device)

    def _sub_terrain_(self): 
        return terrain_utils.SubTerrain(
            width=self.num_rows, 
            length=self.num_cols, 
            vertical_scale=self.config.vertical_scale, 
            horizontal_scale=self.config.horizontal_scale
            )

    def _random_uniform_terrain_(self, config):

        info = [
            config['min_height'],
            config['max_height'],
            0.
        ]

        return info, terrain_utils.random_uniform_terrain(
            self._sub_terrain_(), 
            min_height=config['min_height'], 
            max_height=config['max_height'], 
            step=config['step'], 
            downsampled_scale=config['downsampled_scale']).height_field_raw

    def _flat_terrain_(self, config):
        info = [
            0.,
            0.,
            0.
        ]

        return info, terrain_utils.sloped_terrain(
            self._sub_terrain_(), 
            slope=0.).height_field_raw

    def _sloped_terrain_(self, config):
        info = [
            0.,
            0.,
            config['slope']
        ]

        return info, terrain_utils.sloped_terrain(
            self._sub_terrain_(),
            slope=config['slope']).height_field_raw

    def _pyramid_sloped_terrain_(self, config):
        info = {
            "min": 0.,
            "max": 0.,
            "slope": config['slope']
        }

        self.information_terrain.append(info)

        return terrain_utils.pyramid_sloped_terrain(
            self._sub_terrain_(),
            slope=-config['slope']).height_field_raw

    def _discrete_obstacles_terrain_(self, config):
        info = {
            "min": 0.,
            "max": config['max_height'],
            "slope": 0.
        }

        self.information_terrain.append(info)

        return terrain_utils.discrete_obstacles_terrain(
            self._sub_terrain_(), 
            max_height=config['max_height'], 
            min_size=config['min_size'], 
            max_size=config['max_size'],
            num_rects=config['num_rects']).height_field_raw

    def _wave_terrain_(self, config):
        info = {
            "min": -config['amplitude']/2,
            "max": config['amplitude']/2,
            "slope": 0.
        }

        self.information_terrain.append(info)

        return terrain_utils.wave_terrain(
            self._sub_terrain_(), 
            num_waves=config['num_waves'], 
            amplitude=config['amplitude']).height_field_raw

    def _stairs_terrain_(self, config):
        info = {
            "min": 0.,
            "max": config['step_height'],
            "slope": 45.
        }

        self.information_terrain.append(info)

        return terrain_utils.stairs_terrain(
            self._sub_terrain_(), 
            step_width=config['step_width'],
            step_height=config['step_height']).height_field_raw

    def _pyramid_stairs_terrain_(self, config):
        info = {
            "min": 0.,
            "max": config['step_height'],
            "slope": 45.
        }

        self.information_terrain.append(info)

        return terrain_utils.pyramid_stairs_terrain(
            self._sub_terrain_(), 
            step_width=config['step_width'], 
            step_height=config['step_height']).height_field_raw

    def _stepping_stones_terrain_(self, config):
        info = {
            "min": 0.,
            "max": config['max_height'],
            "slope": 0.
        }

        self.information_terrain.append(info)

        return terrain_utils.stepping_stones_terrain(
            self._sub_terrain_(), 
            stone_size=config['stone_size'],
            stone_distance=config['stone_distance'], 
            max_height=config['max_height'], 
            platform_size=config['platform_size']).height_field_raw

    def build_terrain(self):
 
        heightfield = np.zeros((self.rows*self.num_rows, self.num_cols*self.config.columns), dtype=np.int16)

        for i, dic in enumerate(self.terrain_list):

            name = dic["terrain"]
            info = []

            if not(name in self.terrain_match):
                print(f"Terrain {name} has not been found, flat terrain is included instead")
                name = "flat_terrain"
            
            for j in range(self.config.columns):
                info, heightfield[i*self.num_rows:(i+1)*self.num_rows, j*self.num_cols:(j+1)*self.num_cols] = self.terrain_match[name](dic)

            self.information_terrain.append(info)

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            heightfield, 
            horizontal_scale=self.config.horizontal_scale, 
            vertical_scale=self.config.vertical_scale, 
            slope_threshold=self.config.slope_threshold)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        
        # tm_params.transform.r = gymapi.Quat(0., 0., 0., 0.)
        tm_params.transform.r = gymapi.Quat(0, 0.0, 0.0, 1)
        tm_params.transform.p.x = self.config.border_x
        tm_params.transform.p.y = self.config.border_y - self.config.terrain_length * self.config.columns

        # self.information_terrain = torch.FloatTensor(self.information_terrain).to(self.device)

        return tm_params, vertices, triangles

    def _config_terrain_(self):
        self.num_rows = int(self.config.terrain_width/self.config.horizontal_scale)
        self.num_cols = int(self.config.terrain_length/self.config.horizontal_scale)
