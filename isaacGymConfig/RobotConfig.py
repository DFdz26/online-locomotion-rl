import os
import json
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch
from .BaseConfiguration import BaseConfiguration

from .Rewards import rep_keyword, root_keyword, initial_keyword, projected_gravity_keyword, \
    contact_forces_gravity_keyword, previous_position_keyword
from .Rewards import termination_contact_indices_keyword, penalization_contact_indices_keyword, goal_height_keyword
from .Rewards import foot_contact_indices_keyword, joint_velocity_keyword, foot_velocity_keyword
from .Rewards import base_lin_vel_keyboard, base_ang_vel_keyboard, base_previous_lin_vel_keyboard

import time

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
    # def __init__(self, config_file, env_config, nn, learning_algorithm, logger, rewards, verbose=False):
    def __init__(self, config_file, env_config, rewards, terrain_config=None, curricula=None, verbose=False):
        with open(config_file) as f:
            self.cfg = json.load(f)

        relative_model_folder = self.cfg["asset_options"]["asset_folder"]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.use_gpu = self.cfg["sim_params"]["use_gpu"]
        self.env_config = env_config
        self.terrain_config = terrain_config

        self.asset_root = os.path.join(dir_path, relative_model_folder)
        self.asset_file = self.cfg["asset_options"]["asset_filename"]
        self.asset_name = self.cfg["asset_options"]["asset_name"]
        self.num_envs = self.env_config.num_env
        self.counter_episode = 0
        self.rollout = 0
        self.rep = 0
        self.n_step = 0
        self.curricula = curricula

        self.rewards = rewards
        self.config_intial_position = [self.cfg["asset_options"]["initial_postion"][axis]
                                       for axis in self.cfg["asset_options"]["initial_postion"]]

        if self.env_config.test_joints and self.env_config.test_config.height > 0.:
            self.config_intial_position[2] = self.env_config.test_config.height

        self.starting_training_time = 0
        self.starting_rollout_time = 0

        self.actual_time = 0

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
        self.__create_camera()

        self.root_states = None
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.previous_dof_vel = None
        self.base_quat = None

        self.__prepare_buffers()

        self.default_pose = True
        self.dones = None
        self.reward = None

    def compute_env_distance(self):
        return self.root_states[:, :3] - self.init_root_state[:, :3]

    def get_asset_name(self):
        return self.asset_name
    
    def next_curriculum_level(self):
        pass

    def _create_terrain_(self):

        if self.terrain_config is None:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            self.gym.add_ground(self.sim, plane_params)
        else:
            tm_params, vertices, triangles = self.terrain_config.build_terrain()
            self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def _reset_root(self):

        env_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device,
                             requires_grad=False).nonzero(as_tuple=False).flatten()

        self.root_states[env_ids] = 0.

        self.root_states[env_ids, :3] = self.started_position[env_ids]
        self.previous_robot_position[env_ids, :3] = self.root_states[:, :3].detach().clone()
        self.previous_robot_velocity[env_ids, :3] = 0.

        a = [0., 0., 0., 1]
        self.init_root_state[env_ids, :3] = self.root_states[env_ids, :3]
        self.root_states[env_ids, 3:7] = torch.FloatTensor(a).to(self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.default_pose = True

    def _reset_dofs(self):

        env_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device,
                             requires_grad=False).nonzero(as_tuple=False).flatten()

        self.dof_state[env_ids] = 0.

        for i in range(self.num_envs):
            self.dof_pos[i] = self.default_dof_pos
            self.dof_vel[i] = 0.
            self.previous_dof_vel[i] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_all_envs(self):
        if not(self.curricula is None):
            self.started_position = self.curricula.get_terrain_curriculum(self.started_position)

        self._reset_root()
        self._reset_dofs()

        self.limits = None
        self.finished.fill_(0)
        self.rep = 0

    def check_termination(self):
        if self.limits is None:
            self.limits = self.__check_pos_limit()
        else:
            self.limits += self.__check_pos_limit()

        touching = self.rewards.high_penalization_contacts if hasattr(self.rewards,
                                                                      'high_penalization_contacts') else None
        self.finished |= torch.all(self.limits > 1., dim=0)

        if not (None is touching):
            self.finished |= torch.all(touching > 1., dim=0)


        all_touching = torch.all(touching > 1, dim=0) if not (None is touching) else False
        all_limits = torch.all(self.limits > 1., dim=0)

        self.finished.fill_(0)
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
            goal_height_keyword: self.goal_height,
            foot_contact_indices_keyword: self.feet_indices,
            joint_velocity_keyword: self.dof_vel,
            foot_velocity_keyword: self.foot_velocities,
            base_lin_vel_keyboard: self.base_lin_vel,
            base_ang_vel_keyboard: self.base_ang_vel,
            base_previous_lin_vel_keyboard: self.previous_robot_velocity
        }
        
        return simulation_info

    def compute_final_reward(self):
        simulation_info = self.build_simulation_info()

        rewards = self.rewards.compute_final_reward(simulation_info)
        return rewards

    def post_step(self):
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]

        if not self.env_config.test_joints:

            self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                                  0:3]

            max_length = self.rollout_time / self.env_config.dt
            self.rep += 1

            simulation_info = self.build_simulation_info()

            self.reward = self.rewards.compute_rewards_in_state(simulation_info)
            self.previous_robot_position = self.root_states[:, :3].detach().clone()
            self.previous_robot_velocity = self.base_lin_vel.detach().clone()
            self.check_termination()

    def __prepare_distance_and_termination_rollout_buffers_(self):
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

        self.__prepare_distance_and_termination_rollout_buffers_()
        self.rewards.prepare_buffers()

    def __prepare_sim(self):
        self.gym.prepare_sim(self.sim)

    def controller(self, test_data=None, actions=None, default=False, position_control=True):
        if self.env_config.test_joints:
            actions_scaled = test_data.actions[0, :12] * test_data.scale_actions

            for i in [0, 3, 6, 9]:
                actions_scaled[i] *= test_data.scale_hip

            self.desired_config = self.default_dof_pos + actions_scaled

            self.controller_error = (self.desired_config - self.dof_pos)

            torques = test_data.p_gain * self.controller_error - test_data.d_gain * self.dof_vel

        else:
            if self.env_config.disable_leg:
                actions[:, :3] = 0.

            self.actions = actions
            actions_scaled = actions[:, :12] * self.env_config.actions_scale
            for i in [0, 3, 6, 9]:
                actions_scaled[:, i] *= self.env_config.hip_scale

            self.desired_config = self.default_dof_pos + actions_scaled

            self.controller_error = (self.desired_config - self.dof_pos)

            torques = self.p_gains * self.controller_error - self.d_gains * self.dof_vel

        self.torques = torques

    def __check_pos_limit(self):
        out_of_limits = -(self.dof_pos - self.lower_limit_cuda[:]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.upper_limit_cuda[:]).clip(min=0.)

        return torch.sum(out_of_limits, dim=1)

    def __check_pos_safe(self) -> None:
        dangerous_space = -(self.dof_pos - self.lower_limit_safe[:]).clip(max=0.)
        dangerous_space += (self.dof_pos - self.upper_limit_safe[:]).clip(min=0.)

        return torch.sum(dangerous_space, dim=1)

    def move_dofs(self, test_data, actions=None, position_control=True):
        self.controller(test_data, actions, default=self.default_pose, position_control=position_control)

        if self.env_config.test_joints and self.env_config.joint_to_test > 0:
            print(f"self.desired_config: {self.desired_config[self.env_config.joint_to_test]}, "
                  f"self.dof_pos {self.dof_pos[0][self.env_config.joint_to_test]}, "
                  f"self.torques: {self.torques[0][self.env_config.joint_to_test]}")

        # print(f"torques: {self.torques}")

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def reset_simulation(self):
        self.reset_all_envs()

        obs, _, _, _, _, closed_simulation = self.step(
            None,
            torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        )

        return obs, closed_simulation

    def step(self, test_data=None, actions=None, position_control=True, iterations_without_control=1):

        dones = None
        info = None
        obs = None

        closed_simulation = self.compute_graphics()
        self.previous_dof_vel = self.dof_vel.detach().clone()

        if not closed_simulation:
            for _ in range(iterations_without_control):
                self.move_dofs(test_data, actions, position_control=position_control)

                # step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                self._refresh_gym_tensors_()
            self.post_step()

            obs = self.create_observations()
            dones = self.finished
            info = None

        return obs, self.actions, self.reward, dones, info, closed_simulation

    def compute_graphics(self):
        ending = self.gym.query_viewer_has_closed(self.viewer)

        if not ending:

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        return ending

    def create_observations(self):
        # obs = torch.cat((
        #     self.projected_gravity,
        #     (self.dof_pos - self.default_dof_pos),
        #     self.dof_vel * 0.05,
        #     self.actions),
        #     dim=-1
        # )

        obs = torch.cat((
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * 0.05,
            self.actions,
            self.base_ang_vel * 0.25,
            self.base_lin_vel * 2.0),
            dim=-1
        )

        return {"sensor": obs, "expert": None}

    def __create_camera(self):
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

    def _load_asset(self, verbose=False) -> None:

        if verbose:
            print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root))

        asset_options = gymapi.AssetOptions()
        asset_config_cfg = self.cfg["asset_options"]["asset_config"]
        asset_config_cfg["default_dof_drive_mode"] = convert_drive_mode(self.cfg["asset_options"]["dof_drive_mode"])
        # asset_config_cfg["default_dof_drive_mode"] = 3
        asset_options.disable_gravity = False

        # Load asset options
        asset_options.fix_base_link = asset_config_cfg["fix_base_link"] if not self.env_config.test_joints else True
        asset_options.use_mesh_materials = asset_config_cfg["use_mesh_materials"]
        asset_options.default_dof_drive_mode = asset_config_cfg["default_dof_drive_mode"]
        asset_options.flip_visual_attachments = False

        self.robot_assets = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

        if self.robot_assets is None:
            print("*** Failed to load asset '%s' from '%s'" % (self.asset_file, self.asset_root))
            quit()

        self.num_dof = self.gym.get_asset_dof_count(self.robot_assets)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_assets)
        self.dof_prop_assets = self.gym.get_asset_dof_properties(self.robot_assets)
        self.rigid_shape_assets = self.gym.get_asset_rigid_shape_properties(self.robot_assets)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_assets)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        self.lower_limit = self.dof_prop_assets['lower']

        self.upper_limit = self.dof_prop_assets['upper']

    def _process_rigid_body_props(self, body_prop, n_env):
        # TODO: Changing the mass of the robot

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
            pos_aux_p[1] = i%self.env_config.num_env_colums * self.env_config.spacing_env
            pos_aux_p[0] = math.floor(i/self.env_config.num_env_colums) * self.env_config.spacing_env_x
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

        if not(self.curricula is None):
            self.curricula.set_initial_positions(self.started_position)
