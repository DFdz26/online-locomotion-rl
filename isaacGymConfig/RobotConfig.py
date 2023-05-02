import os
import json
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch
from .BaseConfiguration import BaseConfiguration
from .envConfig import EnvConfig

from .Rewards import rep_keyword, root_keyword, initial_keyword, projected_gravity_keyword, \
    contact_forces_gravity_keyword, previous_position_keyword
from .Rewards import termination_contact_indices_keyword, penalization_contact_indices_keyword, goal_height_keyword
from .Rewards import foot_contact_indices_keyword, joint_velocity_keyword, foot_velocity_keyword
from .Rewards import base_lin_vel_keyboard, base_ang_vel_keyboard, base_previous_lin_vel_keyboard
from .Rewards import previous_actions_keyword, current_actions_keyword, joint_acceleration_keyword
from .Rewards import count_limit_vel_keyword, count_joint_limits_keyword, offset_keyword
import pickle

default_pos = [0.5, 0.32, 0.5] * 4


def convert_drive_mode(mode_str):
    drive_modes = {
        "POS": gymapi.DOF_MODE_POS,
        "FORCE": gymapi.DOF_MODE_EFFORT,
        "VEL": gymapi.DOF_MODE_VEL
    }
    mode = None

    if mode_str in drive_modes:
        mode = drive_modes[mode_str]

    return mode


class RobotConfig(BaseConfiguration):
    def __init__(self, config_file, env_config: EnvConfig, rewards,
                 terrain_config=None, curricula=None, verbose=False):
        with open(config_file) as f:
            self.cfg = json.load(f)

        relative_model_folder = self.cfg["asset_options"]["asset_folder"]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.use_gpu = self.cfg["sim_params"]["use_gpu"]
        self.env_config = env_config
        self.terrain_config = terrain_config
        self.render_GUI = env_config.render_GUI
        self.camera_settings = self.env_config.sensors.Camera
        self.envs_with_camera = None  # Filled once the cameras have been created

        self.asset_root = os.path.join(dir_path, relative_model_folder)
        self.asset_file = self.cfg["asset_options"]["asset_filename"]
        self.asset_name = self.cfg["asset_options"]["asset_name"]
        self.num_envs = self.env_config.num_env
        self.counter_episode = 0
        self.rollout = 0
        self.rep = 0
        self.n_step = 0
        self.curricula = curricula
        self.num_observations = 0
        self.num_observations_sensors = 0
        self.num_expert_observations = 0
        self.viewer = None
        self.torque_limits = None
        self.upper_limits_joint = None
        self.lower_limits_joint = None
        self.surpasing_limits = None
        self.height_samples = None

        self.rewards = rewards
        self.config_intial_position = [self.cfg["asset_options"]["initial_postion"][axis]
                                       for axis in self.cfg["asset_options"]["initial_postion"]]

        if self.env_config.test_joints and self.env_config.test_config.height > 0.:
            self.config_intial_position[2] = self.env_config.test_config.height

        self.starting_training_time = 0
        self.starting_rollout_time = 0

        self.actual_time = 0
        self.start_random_vel = False
        self.randomization_activated = False

        super().__init__(self.cfg, self.env_config.dt)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        self.device = 'cuda:0' if self.use_gpu else 'cpu'

        self.limits = None
        self.finished = None

        self.rollout_time = self.env_config.rollout_time

        self.goal_height = self.cfg["asset_options"]["goal_height"]

        self.default_joint_angles = self.cfg["asset_options"][
            "default_joint_angles"] if self.env_config.default_joints_angles is None else self.env_config.default_joints_angles
        self.default_dof_pos_in = torch.FloatTensor([default_pos] * self.num_envs).to(self.device)

        self.num_dof = None
        self.num_bodies = None
        self.dof_prop_assets = None
        self.rigid_shape_assets = None
        self.robot_assets = None
        self._load_asset(verbose=verbose)

        self.robot_handles = []
        self.envs = []
        self.started_position = None
        self.previous_robot_position = None

        self._create_terrain_()
        self.__create_robot(self.num_envs, verbose=verbose)
        self.__prepare_sim()

        self.viewer = None
        self.sensor_camera = None
        self.recording_in_progress = False
        self.previous_started_position_recording = None
        self.cameras_take_frame = 0
        self.recorded_frames = []

        self.__create_camera()

        self.root_states = None
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.previous_dof_vel = None
        self.aceeleration_dof = None
        self.base_quat = None

        self.__prepare_buffers()

        self.default_pose = True
        self.dones = None
        self.reward = None

        self.save_actions = False
        self.saved_observation = []
        self.saved_actions = []
        self.track_robot = 0

    def enable_random_body_vel_beginning(self):
        self.start_random_vel = True

    def disable_random_body_vel_beginning(self):
        self.start_random_vel = False

    def compute_env_distance(self):
        return self.root_states[:, :3] - self.init_root_state[:, :3]
    
    def get_record_robot(self):
        self.track_robot = self.curricula.get_robot_in_level_zero()
        self.save_actions = True

    def stop_record_robot(self):
        self.save_actions = False

        with open("record_observations_1.pickle", "wb") as f:
            pickle.dump(self.saved_observation, f)

        with open("record_actions_1.pickle", "wb") as f:
            pickle.dump(self.saved_actions, f)

        self.saved_actions = []
        self.saved_observation = []

    def store_information(self, obs, act):
        self.saved_actions.append(act[self.track_robot])
        self.saved_observation.append(obs[self.track_robot])

    def activate_randomization(self):
        self.randomization_activated = True

    def get_asset_name(self):
        return self.asset_name

    def _create_terrain_(self):
        if self.terrain_config is None:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            self.gym.add_ground(self.sim, plane_params)
            self.terr_horizontal_scale = 1.
            self.terr_vertical_scale = 1.
            self.terr_border_x = 0.
            self.terr_border_y = 0.
        else:
            tm_params, vertices, triangles = self.terrain_config.build_terrain()
            self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
            self.terr_horizontal_scale = self.terrain_config.config.horizontal_scale
            self.terr_vertical_scale = self.terrain_config.config.vertical_scale
            self.terr_border_x = self.terrain_config.offset_x
            self.terr_border_y = self.terrain_config.offset_y
            self.height_samples = torch.tensor(self.terrain_config.map_heightfield).view(self.terrain_config.tot_rows,
                                                                                         self.terrain_config.tot_columns).to(self.device)

    def _reset_root(self, env_ids):

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        self.root_states[env_ids] = 0.
        self.root_states[env_ids, :3] = self.started_position[env_ids]

        if self.start_random_vel:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        a = [0., 0., 0., 1]
        self.root_states[env_ids, 3:7] = torch.FloatTensor(a).to(self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.default_pose = True

    def _reset_dofs(self, envs_ids):
        if envs_ids is None:
            envs_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        self.dof_state[envs_ids] = 0.
        self.dof_pos[envs_ids] = self.default_dof_pos
        self.dof_vel[envs_ids] = 0.

        env_ids_int32 = envs_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_all_envs(self):
        if not (self.curricula is None):
            self.started_position = self.curricula.get_terrain_curriculum(self.started_position)

        self.rep = 0

    def _clean_previous_information(self, envs_id):
        self.previous_dof_vel[envs_id] = 0.
        self.aceeleration_dof[envs_id] = 0.
        self.limits[envs_id] = 0.
        self.finished[envs_id] = 0.
        self.surpasing_limits[envs_id] = 0.
        self.surpassing_velocity_limits[envs_id] = 0.
        self.init_root_state[envs_id, :3] = self.root_states[envs_id, :3]
        self.episode_length_buf[envs_id] = 0

        if not(self.previous_actions is None):
            self.previous_actions[envs_id, :] = 0.

    def reset_envs(self, envs_id):

        if envs_id is None:
            envs_id = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        if len(envs_id) == 0:
            return

        self._reset_root(envs_id)
        self._reset_dofs(envs_id)
        self._clean_previous_information(envs_id)

    def check_termination(self):
        if self.limits is None:
            self.limits = self.__check_pos_limit()
        else:
            self.limits += self.__check_pos_limit()

        touching = self.rewards.high_penalization_contacts if hasattr(self.rewards,
                                                                      'high_penalization_contacts') else None
        self.finished |= self.limits > 1.

        if not (None is touching):
            self.finished |= touching > 1.

        all_touching = torch.all(touching > 1, dim=0) if not (None is touching) else False
        all_limits = torch.all(self.limits > 1., dim=0)

        if all_touching or all_limits:
            self.finished.fill_(1)
            print("Touching or limits")

    def _refresh_gym_tensors_(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def build_simulation_info(self):
        simulation_info = {
            rep_keyword: self.rep,
            root_keyword: self.root_states,
            previous_position_keyword: self.previous_robot_position,
            initial_keyword: self.started_position,
            projected_gravity_keyword: self.projected_gravity,
            contact_forces_gravity_keyword: self.contact_forces,
            termination_contact_indices_keyword: self.termination_contact_indices,
            penalization_contact_indices_keyword: self.penalization_contact_indices,
            goal_height_keyword: self.goal_height_individual,
            foot_contact_indices_keyword: self.feet_indices,
            joint_velocity_keyword: self.dof_vel,
            foot_velocity_keyword: self.foot_velocities,
            base_lin_vel_keyboard: self.base_lin_vel,
            base_ang_vel_keyboard: self.base_ang_vel,
            base_previous_lin_vel_keyboard: self.previous_robot_velocity,
            previous_actions_keyword: self.previous_actions,
            current_actions_keyword: self.actions,
            joint_acceleration_keyword: self.aceeleration_dof,
            count_limit_vel_keyword: self.surpassing_velocity_limits,
            count_joint_limits_keyword: self.surpasing_limits,
            # offset_keyword: self.terrain_config.get_height_body_centre(self.root_states[:, :3])
        }

        return simulation_info

    def compute_final_reward(self):
        simulation_info = self.build_simulation_info()

        rewards = self.rewards.compute_final_reward(simulation_info)
        return rewards

    def _init_height_points(self, env_ids):
        """ Returns points at which the height measurments are sampled (in base frame)
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_points, 3)
        """
        y_mesh = [y_point * self.env_config.sensors.HeightMeasurement.y_scale for y_point in self.env_config.sensors.HeightMeasurement.y_mesh]
        x_mesh = [x_point * self.env_config.sensors.HeightMeasurement.x_scale for x_point in self.env_config.sensors.HeightMeasurement.x_mesh]
        y = torch.tensor(y_mesh, device=self.device, requires_grad=False)
        x = torch.tensor(x_mesh, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_points = grid_x.numel()
        points = torch.zeros(len(env_ids), self.num_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        # TODO: change this
        self.num_envs_torch = torch.arange(self.num_envs, device=self.device)
        return points

    def _get_heights(self, env_ids):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises:
            NameError: [description]
        Returns:
            [type]: [description]
        """
        if self.curricula is None:
            return torch.zeros(len(env_ids), self.num_points, device=self.device, requires_grad=False)

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        points[:, :, 0] -= self.terr_border_x
        # points[:, :, 1] -= self.terr_border_y
        points[:, :, 1] -= self.terr_border_y
        # points += 2
        points = (points / self.terr_horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        # print("AAAAAAAAA")
        # print(self.height_samples)
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terr_vertical_scale

    def post_step(self):
        if self.recording_in_progress:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.episode_length_buf += 1

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        
        if self.env_config.sensors.Activations.height_measurement_activated:
            self.measured_heights = self._get_heights(self.num_envs_torch)
            # print(self.height_points)
            # self._draw_debug_vis()

        if not self.env_config.test_joints:
            self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                                  0:3]

            max_length = self.rollout_time / self.env_config.dt
            self.rep += 1

            simulation_info = self.build_simulation_info()

            self.reward = self.rewards.compute_rewards_in_state(simulation_info)
            self.previous_robot_position = self.root_states[:, :3].detach().clone()
            self.previous_robot_velocity = self.base_lin_vel.detach().clone()
            self.previous_actions = self.actions 

            self.check_termination()

        self.recorded_frames = self._get_cameras_frame()
        self.aceeleration_dof = (self.dof_vel - self.previous_dof_vel)/((self.env_config.iterations_without_control + 1) *self.env_config.dt)

    def __prepare_distance_and_termination_rollout_buffers_(self):
        if not self.env_config.test_joints:
            rew = self.rewards.reward_terms

            if not ("x_distance" in rew):
                rew["x_distance"] = {
                    "weight": 0.,
                    "reward_data": {
                        "absolute_distance": False
                    }
                }

            if not ("high_penalization_contacts" in rew):
                rew["high_penalization_contacts"] = {
                    "weight": 0.,
                    "reward_data": {
                        "max_clip": 0.0,
                        "weights": {
                            "correction_state": 0.
                        },
                    }
                }

            self.rewards.change_rewards(rew)

    def __prepare_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.init_dof_state = self.dof_state.detach().clone()
        self.init_root_state = self.root_states.detach().clone()
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.base_pos = self.root_states[:, :3]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat(
            (self.num_envs, 1))

        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.p_gains.fill_(25.)
        self.d_gains.fill_(.5)

        for i in range(self.num_dof):
            name = self.dof_names[i]

            found = False
            for dof_name in self.cfg["asset_options"]["gains_constants"]["Kp"].keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg["asset_options"]["gains_constants"]["Kp"][dof_name]
                    self.d_gains[i] = self.cfg["asset_options"]["gains_constants"]["Kd"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")

        body_names = self.gym.get_asset_rigid_body_names(self.robot_assets)

        termination_contact_names = []
        for name in self.cfg["asset_options"]["terminate_after_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])

        penalization_contact_names = []
        for name in self.cfg["asset_options"]["penalization_contacts_on"]:
            penalization_contact_names.extend([s for s in body_names if name in s])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        self.penalization_contact_indices = torch.zeros(len(penalization_contact_names), dtype=torch.long,
                                                        device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.robot_handles[0],
                                                                                        termination_contact_names[i])

        for i in range(len(penalization_contact_names)):
            self.penalization_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                         self.robot_handles[0],
                                                                                         penalization_contact_names[i])

        feet_names = [s for s in body_names if self.cfg["asset_options"]["foot_contacts_on"] in s]

        print(body_names)
        print(feet_names)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                         self.robot_handles[0],
                                                                         feet_names[i])

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]

        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.previous_dof_vel = self.dof_vel.detach().clone()
        self.aceeleration_dof = self.dof_vel.detach().clone()

        self.__prepare_distance_and_termination_rollout_buffers_()

        if not self.env_config.test_joints:
            self.rewards.prepare_buffers()

        self.actions = None
        self.previous_actions = None

        self.goal_height_individual = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.goal_height_individual.fill_(self.goal_height)

        self.torque_limits = torch.tensor(self.cfg["asset_options"]["torque_limits"], requires_grad=False).to(self.device)
        self.surpasing_limits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.surpassing_velocity_limits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)

        if self.env_config.sensors.Activations.height_measurement_activated:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device))
        self.measured_heights = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)

    def __get_body_randomization_information(self, get_mass):
        if self.curricula is None:
            return None

        if not self.curricula.randomization_available():
            return None

        fri_aux, res_aux, payload_aux = self.curricula.get_randomized_body_properties(self.num_envs,
                                                                                      include_mass=get_mass)

        if not(res_aux is None):
            self.restitutions = fri_aux.detach().clone()

        if not(fri_aux is None):
            self.friction_coeffs = fri_aux.detach().clone()

        if not(payload_aux is None):
            self.payloads = payload_aux.detach().clone()

    def get_new_randomization(self):
        self.__get_motors_randomization_information()

    def __get_motors_randomization_information(self):
        if self.curricula is None:
            return None

        if not self.curricula.randomization_available():
            return None

        kp_aux, kd_aux, mnotor_strength_aux = self.curricula.get_randomized_motor_properties(self.num_envs)

        

        if not(mnotor_strength_aux is None):

            self.motor_strengths = mnotor_strength_aux.squeeze().detach().clone()
            self.motor_strengths = torch.add(torch.div(self.motor_strengths, 100), 1.)

    def __prepare_sim(self):
        self.gym.prepare_sim(self.sim)

    def controller(self, test_data=None, actions=None, default=False, position_control=True):
        if self.print_flag_:
            print(actions)

        if self.env_config.test_joints:
            test_data.actions = test_data.actions * self.mirrored

            actions_scaled = test_data.actions[:, :12] * test_data.scale_actions
            actions_scaled[:, [0, 3, 6, 9]] *= test_data.scale_hip

            self.desired_config = self.default_dof_pos + actions_scaled

            self.controller_error = (self.desired_config - self.dof_pos)

            torques = test_data.p_gain * self.controller_error - test_data.d_gain * self.dof_vel

        else:
            actions = actions * self.mirrored

            if self.env_config.disable_leg:
                actions[:, :3] = 0.

            self.actions = actions
            actions_scaled = actions[:, :12] * self.env_config.actions_scale
            actions_scaled[:, [0, 3, 6, 9]] *= self.env_config.hip_scale

            self.desired_config = self.default_dof_pos + actions_scaled
            self.surpasing_limits = torch.sum(self.desired_config.ge(self.upper_limits_joint), dim=-1)
            self.surpasing_limits |= torch.sum(self.desired_config.le(self.lower_limits_joint), dim=-1)
            
            self.desired_config.clip(min=self.lower_limits_joint, max=self.upper_limits_joint)

            self.controller_error = (self.desired_config - self.dof_pos)

            torques = self.p_gains * self.controller_error - self.d_gains * self.dof_vel

            if not(self.curricula is None):
                torques = torques * self.motor_strengths.unsqueeze(1)

        self.torques = torques.clip(max=self.torque_limits, min=-self.torque_limits)

    def __check_pos_limit(self):
        out_of_limits = -(self.dof_pos - self.lower_limit_cuda[:]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.upper_limit_cuda[:]).clip(min=0.)

        return torch.sum(out_of_limits, dim=1)

    def __check_pos_safe(self) -> None:
        dangerous_space = -(self.dof_pos - self.lower_limit_safe[:]).clip(max=0.)
        dangerous_space += (self.dof_pos - self.upper_limit_safe[:]).clip(min=0.)

        return torch.sum(dangerous_space, dim=1)
    
    def _check_velocity_limits(self):
        self.dof_vel
        self.surpassing_velocity_limits = torch.sum(self.dof_vel.ge(self.velocity_limits), dim=-1)
        self.surpassing_velocity_limits |= torch.sum(self.dof_vel.le(-self.velocity_limits), dim=-1)
        
        self.dof_vel.clip(min=-self.velocity_limits, max=self.velocity_limits)

    def move_dofs(self, test_data, actions=None, position_control=True):
        self._check_velocity_limits()
        self.controller(test_data, actions, default=self.default_pose, position_control=position_control)

        if self.env_config.test_joints and self.env_config.joint_to_test > 0:
            print(f"self.desired_config: {self.desired_config[self.env_config.joint_to_test]}, "
                  f"self.dof_pos {self.dof_pos[0][self.env_config.joint_to_test]}, "
                  f"self.torques: {self.torques[0][self.env_config.joint_to_test]}")

        # print(f"torques: {self.torques}")

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        

    def reset_simulation(self):
        self.reset_all_envs()

        obs, obs_expert, _, _, _, _, closed_simulation = self.step(
            None,
            torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        )

        return obs, obs_expert, closed_simulation

    def step(self, test_data=None, actions=None, position_control=True, iterations_without_control=1):

        dones = None
        info = None
        obs = None
        obs_expert = None
        closed_simulation = self.compute_graphics()

        if not self.env_config.test_joints:            
            self.previous_dof_vel = self.dof_vel.detach().clone()
            actions = torch.clip(actions, -self.env_config.clip_actions, self.env_config.clip_actions).to(self.device)

        if not closed_simulation:
            for _ in range(self.env_config.iterations_without_control + 1):
                self.move_dofs(test_data, actions, position_control=position_control)

                # step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                self._refresh_gym_tensors_()
            self.post_step()

            if not self.env_config.test_joints:

                obs, obs_expert = self.create_observations()
                dones = self.finished
                info = None

            if self.save_actions:
                self.store_information(obs, self.desired_config)

        return obs, obs_expert, self.actions, self.reward, dones, info, closed_simulation

    def compute_graphics(self):
        if self.render_GUI:
            ending = self.gym.query_viewer_has_closed(self.viewer)

            if not ending:

                if self.device != 'cpu':
                    self.gym.fetch_results(self.sim, True)

                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            return ending
        else:
            return False

    def get_num_observations(self):
        return self.num_observations, self.num_observations_sensors, self.num_expert_observations
    
    def print_flag_act(self):
        self.print_flag_ = True

    def create_observations(self):
        # obs = torch.cat((
        #     self.projected_gravity,
        #     (self.dof_pos - self.default_dof_pos),
        #     self.dof_vel * 0.05,
        #     self.actions),
        #     dim=-1
        # )

        # obs = torch.cat((
        #     self.projected_gravity,
        #     (self.dof_pos - self.default_dof_pos),
        #     self.dof_vel * 0.05,
        #     self.actions,
        #     self.base_ang_vel * 0.25,
        #     self.base_lin_vel * 2.0),
        #     dim=-1
        # )

        obs = torch.cat((
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * 0.05,
            self.base_ang_vel * 0.25,
            self.actions),
            dim=-1
        )

        obs = torch.clip(obs, -self.env_config.clip_observations, self.env_config.clip_observations)

        # expert = torch.cat((
        #     torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)),
        #     dim=-1
        # )

        in_contact = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.).to(torch.float32)
        # contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        contact_forces = self.contact_forces[:, self.feet_indices, :].squeeze(dim=-1)

        # print(self.contact_forces[:, self.feet_indices, :].size())
        h = self.root_states[:, 2].unsqueeze(dim=-1)

        # expert = torch.cat((
        #     self.terrain_config.get_info_terrain(self.base_pos),
        #     in_contact),
        #     dim=-1
        # )

        expert = torch.cat((
            contact_forces[:, :, 0]/50 - 1,
            contact_forces[:, :, 1]/50 - 1,
            contact_forces[:, :, 2]/50 - 1,
            h,
            in_contact,
            self.base_lin_vel * 2.0),
            dim=-1
        )

        if self.env_config.sensors.Activations.height_measurement_activated:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * 5.0
            expert = torch.cat(
                (expert,
                 heights),
                dim=-1
            )

        # scale all the randomization from -1 to 1
        scales_shift = None

        if not (self.curricula is None):
            scales_shift = self.curricula.get_scales_shift_randomized_parameters(default_motor_strength=1.)

        if not (scales_shift is None):
            # scale all the randomization from -1 to 1
            friction_coeffs_scale, friction_coeffs_shift = scales_shift["friction"]
            restitutions_scale, restitutions_shift = scales_shift["restitutions"]
            payloads_scale, payloads_shift = scales_shift["payloads"]
            motor_strengths_scale, motor_strengths_shift = scales_shift["motor_strengths"]

            e_fri = self.env_config.cfg_observations.enable_observe_friction
            e_res = self.env_config.cfg_observations.enable_observe_restitution
            e_pay = self.env_config.cfg_observations.enable_observe_payload
            e_mot = self.env_config.cfg_observations.enable_observe_motor_strength

            randomized_obs = torch.cat(
                ((self.friction_coeffs - friction_coeffs_shift).unsqueeze(1) * friction_coeffs_scale * int(e_fri),  # friction coeff
                 (self.restitutions - restitutions_shift).unsqueeze(1) * restitutions_scale * int(e_res),  # friction coeff
                 (self.payloads - payloads_shift).unsqueeze(1) * payloads_scale * int(e_pay),  # payload
                 (self.motor_strengths - motor_strengths_shift).unsqueeze(1) * motor_strengths_scale * int(e_mot),  # motor strength
                 ), dim=-1)

            expert = torch.cat((
                randomized_obs,
                expert
            ), dim=-1)

        # expert = in_contact

        expert = torch.clip(expert, -self.env_config.clip_observations, self.env_config.clip_observations)

        self.num_observations_sensors = obs.size()[1]
        self.num_expert_observations = expert.size()[1]
        self.num_observations = self.num_observations_sensors + self.num_expert_observations

        return obs, expert

    def _create_default_camera(self, n_env):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_settings.width
        camera_props.height = self.camera_settings.height
        camera_handle = self.gym.create_camera_sensor(self.envs[n_env], camera_props)

        if self.envs_with_camera is None:
            self.envs_with_camera = []

        self.envs_with_camera.append(n_env)  # Keep a record the env with cameras

        return camera_handle

    def stop_recording_videos(self):
        self.started_position[self.envs_with_camera] = self.previous_started_position_recording
        self.recording_in_progress = False

    def set_up_recording_video(self, terrains_to_record: list):
        """
        Function to set up the needed configurations for starting recording the simulation.
        Receives a list with the desired terrains to record. If the length of the list exceeds the number of
        available cameras, the last terrains will be recorded
        :param terrains_to_record: List with the terrains that want to be recorded.
        :return:
        """

        self.previous_started_position_recording = self.started_position[self.envs_with_camera]
        self.recording_in_progress = True

        if self.curricula is None or terrains_to_record is None:
            self.cameras_take_frame = 1
            return None

        n_terrains = len(terrains_to_record)

        if n_terrains > len(self.envs_with_camera):
            self.cameras_take_frame = len(self.envs_with_camera)
            desired_terrains = terrains_to_record[-self.cameras_take_frame:]
        else:
            desired_terrains = terrains_to_record
            self.cameras_take_frame = n_terrains

        for i in range(self.cameras_take_frame):
            selected_env = self.envs_with_camera[i]

            self.started_position[selected_env] = self.curricula.jump_env_to_terrain(
                selected_env,
                desired_terrains[i],
                self.started_position[selected_env]
            )

            if desired_terrains[i] == 0:
                self.started_position[selected_env] = self.previous_started_position_recording[i]

    def _get_cameras_frame(self):
        if not self.recording_in_progress:
            return []

        frames = []

        for i in range(self.cameras_take_frame):
            frames.append(self._get_frame_individual_camera(self.sensor_camera[i], self.envs_with_camera[i]))

        return frames

    def _get_frame_individual_camera(self, camera, selected_env):
        bx, by, bz = self.root_states[selected_env, 0], self.root_states[selected_env, 1], self.root_states[
            selected_env, 2]
        self.gym.set_camera_location(camera, self.envs[selected_env], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        frame = self.gym.get_camera_image(self.sim, self.envs[selected_env], camera, gymapi.IMAGE_COLOR)
        frame = frame.reshape((self.camera_settings.height, self.camera_settings.width, 4))

        return frame

    def __create_camera(self):
        if self.render_GUI:
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            pos = [-5, 0, 1]  # [m]
            lookat = [11., 5, 3.]  # [m]

            cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

            self.viewer = viewer

            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()

        if self.env_config.sensors.Activations.camera_activated:
            self.sensor_camera = []

            n_camera_settings = self.env_config.sensors.Camera.n_camera
            n_cameras = n_camera_settings if self.num_envs > n_camera_settings else self.num_envs

            for env in range(n_cameras):
                self.sensor_camera.append(self._create_default_camera(env))

    def _load_asset(self, verbose=False) -> None:

        if verbose:
            print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root))

        asset_options = gymapi.AssetOptions()
        print("-------")
        print(f"{asset_options.__dir__()}")
        print("-------")
        asset_config_cfg = self.cfg["asset_options"]["asset_config"]
        asset_config_cfg["default_dof_drive_mode"] = convert_drive_mode(self.cfg["asset_options"]["dof_drive_mode"])
        # asset_config_cfg["default_dof_drive_mode"] = 3
        asset_options.disable_gravity = False

        # Load asset options
        asset_options.fix_base_link = asset_config_cfg["fix_base_link"] if not self.env_config.test_joints else True
        asset_options.use_mesh_materials = asset_config_cfg["use_mesh_materials"]
        asset_options.default_dof_drive_mode = asset_config_cfg["default_dof_drive_mode"]
        asset_options.density = 0.01
        asset_options.armature = 0.
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.flip_visual_attachments = False
        asset_options.replace_cylinder_with_capsule  = True
        asset_options.collapse_fixed_joints  = True

        self.robot_assets = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

        if self.robot_assets is None:
            print("*** Failed to load asset '%s' from '%s'" % (self.asset_file, self.asset_root))
            quit()

        self.num_dof = self.gym.get_asset_dof_count(self.robot_assets)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_assets)
        self.dof_prop_assets = self.gym.get_asset_dof_properties(self.robot_assets)
        self.rigid_shape_assets = self.gym.get_asset_rigid_shape_properties(self.robot_assets)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_assets)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_assets)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        self.lower_limit = self.dof_prop_assets['lower']
        self.velocity_limits = torch.tensor(self.dof_prop_assets['velocity'], requires_grad=False).to(self.device)
        print(self.velocity_limits)
        self.upper_limit = self.dof_prop_assets['upper']
        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self.upper_limits_joint = torch.tensor(self.dof_prop_assets['upper'], requires_grad=False).to(self.device)
        self.lower_limits_joint = torch.tensor(self.dof_prop_assets['lower'], requires_grad=False).to(self.device)

        self.mirrored = torch.ones(self.num_dof, dtype=torch.int, device=self.device, requires_grad=False)

        if 'mirrored' in asset_config_cfg:
            if len(asset_config_cfg['mirrored']) != self.num_dof:
                raise Exception("The number of the mirrored dof in configuration is not the same as the number of " \
                                f'dof. Number of dof: {self.num_dof}')
            mirrored = [-2 * i for i in asset_config_cfg['mirrored']]
            self.mirrored += torch.IntTensor(mirrored).to(self.device)
            # self.mirrored = -1 * int(asset_config_cfg['mirrored']) + 1 * int(not asset_config_cfg['mirrored'])
            # self.mirrored = torch.IntTensor(self.mirrored, device=self.device, requires_grad=False)
    

        self.print_flag_ = False


    def _process_rigid_body_props(self, body_prop, n_env):
        self.default_body_mass = body_prop[0].mass

        body_prop[0].mass = self.default_body_mass + self.payloads[n_env]

        return body_prop

    def __create_robot(self, num_robots, verbose=False):
        spacing = 0.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        pose = gymapi.Transform()
        # pos_aux_p = [0., 0., 0.68]
        pos_aux_p = self.config_intial_position
        pose.p = gymapi.Vec3(0., 0., 0.8)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        self.ranges = torch.from_numpy(self.upper_limit - self.lower_limit).to(self.device)
        self.mids = torch.from_numpy(0.5 * (self.upper_limit + self.lower_limit)).to(self.device)
        self.lower_limit_cuda = torch.from_numpy(self.lower_limit).to(self.device)
        self.upper_limit_cuda = torch.from_numpy(self.upper_limit).to(self.device)
        self.upper_limit_safe = (self.mids + self.upper_limit_cuda) * 0.5
        self.lower_limit_safe = (self.mids + self.lower_limit_cuda) * 0.5

        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.motor_strengths = torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        
        self.restitutions_default = self.restitutions.detach().clone()
        self.friction_coeffs_default = self.friction_coeffs.detach().clone()

        self.__get_body_randomization_information(True)

        default_dof_state = np.zeros(self.num_dof, gymapi.DofState.dtype)
        # default_dof_state["pos"] = self.mids
        default_dof_state["pos"] = self.default_dof_pos.cpu().numpy()

        num_per_row = int(math.sqrt(num_robots))
        self.started_position = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.previous_robot_position = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.previous_robot_velocity = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        if verbose:
            print("Creating %d environments" % num_robots)

        for i in range(num_robots):
            # Create one environment

            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            pos_aux_p[1] = i % self.env_config.num_env_colums * self.env_config.spacing_env
            pos_aux_p[0] = math.floor(i / self.env_config.num_env_colums) * self.env_config.spacing_env_x
            self.started_position[i] = torch.FloatTensor(pos_aux_p).to(self.device)
            self.previous_robot_position[i] = self.started_position[i]
            pose.p = gymapi.Vec3(*pos_aux_p)
            robot_handle = self.gym.create_actor(env, self.robot_assets, pose, self.asset_name, i,
                                                 0 if self.cfg["asset_options"]["asset_config"][
                                                     "self_collision"] else 1, 0)
            self.gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

            dof_props = self.dof_prop_assets

            self.gym.set_actor_dof_properties(env, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)

            self.gym.set_actor_rigid_body_properties(env, robot_handle, body_props, recomputeInertia=True)

            self.envs.append(env)
            self.robot_handles.append(robot_handle)

        if not (self.curricula is None):
            self.curricula.set_initial_positions(self.started_position)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def __del__(self):
        if not (self.viewer is None):
            self.gym.destroy_viewer(self.viewer)

        return super().__del__()


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)
