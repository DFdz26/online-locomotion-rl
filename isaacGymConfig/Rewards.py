import torch
import time

rep_keyword = 'rep'
root_keyword = 'root_state'
offset_keyword = 'offset_height'
base_lin_vel_keyboard = 'base_lin_vel'
base_previous_lin_vel_keyboard = 'base_previous_lin_vel'
base_ang_vel_keyboard = 'base_ang_vel'
previous_position_keyword = 'previous_position'
initial_keyword = 'initial_state'
projected_gravity_keyword = 'projected_gravity'
contact_forces_gravity_keyword = 'contact_forces'
termination_contact_indices_keyword = 'termination_contact_indices'
penalization_contact_indices_keyword = 'penalization_contact_indices'
foot_contact_indices_keyword = 'foot_contact_indices'
goal_height_keyword = 'goal_height'
joint_velocity_keyword = 'joint_velocity'
foot_velocity_keyword = 'feet_velocity'
previous_joint_velocity_keyboard = 'previous_joint_velocity'
dt_simulation_keyboard = 'dt_sim'
previous_actions_keyword = 'previous_action'
current_actions_keyword = 'current_action'
joint_acceleration_keyword = 'acceleration_joints'
count_limit_vel_keyword = 'velocity_limits_count'
count_joint_limits_keyword = 'joint_limits_count'
current_torques_keyword = 'current_torques'


class IndividualReward:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.x_distance = None
        self.y_distance = None

        self.iteration = 0

    ###############################################################################################

    """
    Preparation of the buffers
    """
    def _prepare_buffer_velocity_smoothness_term_(self):
        self.vel_smothness_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        
    def _prepare_buffer_limits_term_(self):
        self.limits_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                       requires_grad=False)

    def _prepare_buffer_smoothness_term_(self):
        self.prev_acc = None
        self.prev_speed = None
        self.jerk_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                       requires_grad=False)

    def _prepare_buffer_z_vel_term_(self):
        self.accumulative_error_z_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)

    def _prepare_buffer_y_velocity_term_(self):
        self.accumulative_error_y_velocity = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                         requires_grad=False)

    def _prepare_buffer_x_velocity_term_(self):
        self.accumulative_error_x_velocity = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                         requires_grad=False)

    def _prepare_buffer_roll_pitch_term_(self):
        self.accumulative_error_roll_pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                         requires_grad=False)

    def _prepare_buffer_yaw_vel_term_(self):
        self.accumulative_error_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                  requires_grad=False)

    def _prepare_buffer_slippery_term_(self):
        self.slippery_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                           requires_grad=False)

    def _prepare_buffer_x_distance_term_(self):
        pass

    def _prepare_buffer_vel_cont_term_(self):
        self.accumulative_vel_cont_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                       requires_grad=False)

    def _prepare_buffer_vibration_term_(self):
        self.accumulative_vibration_term = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                       requires_grad=False)

    def _prepare_buffer_y_distance_term_(self):
        pass

    def _prepare_buffer_speed_error_term_(self):
        self.accumulative_error_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                    requires_grad=False)

    def _prepare_buffer_stability_term_(self):
        self.accumulative_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                               requires_grad=False)
        self.accumulative_height_2 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.acc = torch.zeros([3, self.num_envs], dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_buffer_low_penalization_contacts_term_(self):
        self.low_penalization_contacts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device,
                                                     requires_grad=False)

    def _prepare_buffer_high_penalization_contacts_term_(self):
        self.high_penalization_contacts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device,
                                                      requires_grad=False)
        self.high_penalization_state = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device,
                                                      requires_grad=False)

    def _prepare_buffer_height_error_term_(self):
        self.accumulative_height_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                     requires_grad=False)

    def _prepare_buffer_orthogonal_angle_error_term_(self):
        self.orthogonal_angle_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        
    def _prepare_buffer_changed_actions_term_(self):
        self.changed_actions_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        
    def _prepare_buffer_torque_penalization_term_(self):
        self.torque_penalization_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)

    ###############################################################################################
    """
    Compute the needed calculations in every step
    """
    def _compute_in_state_changed_actions_term_(self, simulation_info, reward_data):
        previous_action = simulation_info[previous_actions_keyword]
        current_action = simulation_info[current_actions_keyword]
        weight = reward_data['weight']

        if previous_action is None:
            return 0

        difference = torch.norm((previous_action - current_action), dim=-1) * weight

        self.changed_actions_buffer += difference 
        return difference
    
    def _compute_in_state_limits_term_(self, simulation_info, reward_data):
        count_limit_vel = simulation_info[count_limit_vel_keyword]
        count_limit_joint = simulation_info[count_joint_limits_keyword]

        w_join = reward_data["joint_limits"]
        w_vel = reward_data["velocity_limits"]
        w_tot = reward_data["weight"]

        in_state = w_tot*(w_join * count_limit_joint + w_vel * count_limit_vel)
        self.limits_buffer += in_state

        return in_state
    

    def _compute_in_state_torque_penalization_term_(self, simulation_info, reward_data):
        torque = simulation_info[current_torques_keyword]
        weight = reward_data['weight']

        in_state = torch.sum(torch.square(torque), dim=-1) * weight

        self.torque_penalization_buffer += in_state
        return in_state
    
    def _compute_in_state_velocity_smoothness_term_(self, simulation_info, reward_data):
        velocity = simulation_info[joint_velocity_keyword]
        acceleration = simulation_info[joint_acceleration_keyword]
        w_acc = reward_data["weight_acc"]
        w_vel = reward_data["weight_vel"]
        w_tot = reward_data["weight"]

        vel = torch.square(velocity)
        acc = torch.square(acceleration)
        
        in_state = w_tot * torch.sum((w_vel * torch.square(vel) + w_acc * torch.square(acc)), dim=-1)
        self.vel_smothness_buffer += in_state

        return in_state

    def _compute_in_state_x_distance_term_(self, simulation_info, reward_data):
        previous_position = simulation_info[previous_position_keyword][:, 0]
        actual_position = simulation_info[root_keyword][:, 0]

        self.x_distance_step = actual_position - previous_position
        return self.x_distance_step

    def _compute_in_state_slippery_term_(self, simulation_info, reward_data):
        # Check if any foot is in contact with the ground
        contact_forces = simulation_info[contact_forces_gravity_keyword]
        foot_indices = simulation_info[foot_contact_indices_keyword]
        current_foot_vel = simulation_info[foot_velocity_keyword]
        slipperiness_penalty_coef = reward_data["slippery_coef"]

        in_contact = torch.norm(contact_forces[:, foot_indices, :], dim=-1) > 1.

        # Penalize the slipperiness of each foot that is in contact with the ground
        in_state_slipperiness_penalty = slipperiness_penalty_coef * torch.square(
            torch.sum(torch.norm(current_foot_vel, dim=-1) * in_contact, dim=-1))

        self.slippery_buffer += in_state_slipperiness_penalty

        return in_state_slipperiness_penalty

    def _compute_in_state_smoothness_term_(self, simulation_info, reward_data):

        speed = simulation_info[joint_velocity_keyword]
        jerk_coef = reward_data['jerk_coef']
        dt = simulation_info[dt_simulation_keyboard]

        if self.prev_speed is None:
            self.prev_speed = speed.detach().clone()
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        acceleration = (speed - self.prev_speed) / dt
        jerk = torch.abs(acceleration - self.prev_acc) if self.prev_acc is not None else torch.abs(acceleration)
        self.prev_acc = acceleration

        self.prev_speed = speed.detach().clone()
        int_state = jerk_coef * torch.sum(jerk, dim=-1)

        self.jerk_buffer += int_state
        return int_state

    def _compute_in_state_vibration_term_(self, simulation_info, reward_data):
        dt_sim = simulation_info[dt_simulation_keyboard]
        previous_dof_vel = simulation_info[previous_joint_velocity_keyboard]
        actual_dof_vel = simulation_info[joint_velocity_keyword]

        acceleration_dof_sq = torch.sum(torch.square((previous_dof_vel - actual_dof_vel) / dt_sim), dim=1)
        in_state = torch.sum(0.01 * torch.square(actual_dof_vel) + acceleration_dof_sq, dim=1)

        self.accumulative_vibration_term += in_state
        return in_state

    def _compute_in_state_y_distance_term_(self, simulation_info, reward_data):
        previous_position = simulation_info[previous_position_keyword][:, 1]
        actual_position = simulation_info[root_keyword][:, 1]

        self.y_distance_step = actual_position - previous_position
        return self.y_distance_step

    @staticmethod
    def __error_in_speed_int(actual, desired, compute, division_):
        if compute:
            error = actual - desired
            return torch.exp(-torch.torch.square(error) / division_)
        else:
            return 0

    def _compute_in_state_speed_error_term_(self, simulation_info, reward_data):

        base_vel = simulation_info[base_lin_vel_keyboard]
        speed_in_x_rw = reward_data['speed_in_x']
        speed_in_y_rw = reward_data['speed_in_y']
        speed_in_z_rw = reward_data['speed_in_z']
        division = reward_data['division']

        error_z = self.__error_in_speed_int(speed_in_z_rw, base_vel[:, 2], not (None is speed_in_z_rw), division)
        error_x = self.__error_in_speed_int(speed_in_x_rw, base_vel[:, 0], not (None is speed_in_x_rw), division)
        error_y = self.__error_in_speed_int(speed_in_y_rw, base_vel[:, 1], not (None is speed_in_y_rw), division)

        in_state = error_z + error_x + error_y
        self.accumulative_error_speed += in_state

        return in_state

    def _compute_in_state_stability_term_(self, simulation_info, reward_data):
        projected_gravity = simulation_info[projected_gravity_keyword]
        root_state = simulation_info[root_keyword]

        error = 0
        for i in range(3):
            self.acc[i] += projected_gravity[:, i].detach()
            error += self.__error_in_speed_int(projected_gravity[:, i], 0, True, 0.5)

        self.accumulative_height += root_state[:, 2]
        self.accumulative_height_2 += torch.square(root_state[:, 2])

        return -error

    def _compute_in_state_low_penalization_contacts_term_(self, simulation_info, reward_data):
        contact_forces = simulation_info[contact_forces_gravity_keyword]
        penalization_contact_indices = simulation_info[penalization_contact_indices_keyword]

        in_state = torch.any(torch.norm(contact_forces[:, penalization_contact_indices, :], dim=-1) > 1., dim=1)
        self.low_penalization_contacts += in_state

        return in_state

    def _compute_in_state_high_penalization_contacts_term_(self, simulation_info, reward_data):
        contact_forces = simulation_info[contact_forces_gravity_keyword]
        termination_contact_indices = simulation_info[termination_contact_indices_keyword]

        in_state = torch.any(torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1) > 1., dim=1)
        # print(in_state)
        self.high_penalization_contacts += in_state
        self.high_penalization_state = in_state

        return in_state

    def _compute_in_state_height_error_term_(self, simulation_info, reward_data):
        z_state = simulation_info[root_keyword][:, 2]
        goal_height = simulation_info[goal_height_keyword]

        in_state = torch.sqrt(torch.abs(z_state - goal_height)) / 100
        self.accumulative_height_error += in_state

        return in_state

    def _compute_in_state_z_vel_term_(self, simulation_info, reward_data):
        weight = reward_data["weight"]
        exponential = reward_data["exponential"]

        z_vel = simulation_info[base_lin_vel_keyboard][:, 2]

        if exponential:
            divider = reward_data["divider"]
            in_state = weight * self.__error_in_speed_int(z_vel, 0, True, divider)
        else:
            in_state = weight * torch.square(z_vel)

        self.accumulative_error_z_speed += in_state

        return in_state

    def _compute_in_state_roll_pitch_term_(self, simulation_info, reward_data):
        weight = reward_data["weight"]
        exponential = reward_data["exponential"]

        x_ang_vel = simulation_info[base_ang_vel_keyboard][:, 0]
        y_ang_vel = simulation_info[base_ang_vel_keyboard][:, 1]

        if exponential:
            divider = reward_data["divider"]
            in_state = weight * self.__error_in_speed_int(x_ang_vel, 0, True, divider)
            in_state += weight * self.__error_in_speed_int(y_ang_vel, 0, True, divider)
        else:
            in_state = (1.25 * weight) * torch.abs(x_ang_vel) + weight * torch.abs(y_ang_vel)

        self.accumulative_error_roll_pitch += in_state

        return in_state

    def _compute_in_state_yaw_vel_term_(self, simulation_info, reward_data):
        weight = reward_data["weight"]
        command = reward_data["command"]
        exponential = reward_data["exponential"]

        z_ang_vel = simulation_info[base_ang_vel_keyboard][:, 2]

        if exponential:
            divider = reward_data["divider"]
            in_state = weight * self.__error_in_speed_int(z_ang_vel, command, True, divider)
        else:
            in_state = weight * torch.abs(z_ang_vel - command)

        self.accumulative_error_yaw += in_state

        return in_state

    def _compute_in_state_y_velocity_term_(self, simulation_info, reward_data):
        y_vel = simulation_info[base_lin_vel_keyboard][:, 1]
        weight = reward_data["weight"]

        exponential = reward_data["exponential"]

        if exponential:
            divider = reward_data["divider"]
            in_state = weight * self.__error_in_speed_int(y_vel, 0, True, divider)
        else:
            in_state = weight * torch.abs(y_vel)

        self.accumulative_error_y_velocity += in_state

        return in_state

    def _compute_in_state_x_velocity_term_(self, simulation_info, reward_data):
        x_vel = simulation_info[base_lin_vel_keyboard][:, 0]
        weight = reward_data["weight"]

        # if self.iteration % 10 == 0:
        #     print(f"x_vel: {x_vel}")

        # self.iteration += 1
        # self.iteration %+ 10

        exponential = reward_data["exponential"]

        if exponential:
            divider = reward_data["divider"]
            goal = reward_data["goal"]
            in_state = weight * self.__error_in_speed_int(x_vel, goal, True, divider)
        else:
            in_state = weight * x_vel

        self.accumulative_error_x_velocity += in_state

        return in_state

    def _compute_in_state_torque_term_(self, simulation_info, reward_data):
        pass

    def _compute_in_state_vel_cont_term_(self, simulation_info, reward_data):
        x_vel = simulation_info[base_lin_vel_keyboard][:, 0]
        previous_vel = simulation_info[base_previous_lin_vel_keyboard][:, 0]

        in_state = torch.sqrt(torch.abs(x_vel - previous_vel))
        self.accumulative_vel_cont_error += in_state

        return in_state

    def _compute_in_state_orthogonal_angle_error_term_(self, simulation_info, reward_data):
        orientation = simulation_info[projected_gravity_keyword]
        weight = reward_data["weight"]

        in_state = weight * torch.sqrt(torch.square(orientation[:, 0]) + torch.square(orientation[:, 1]))
        self.accumulative_vel_cont_error += in_state

        return in_state

    ###############################################################################################

    def _compute_final_x_distance_term_(self, simulation_info, reward_data):

        if self.x_distance is None:
            self.x_distance = simulation_info[root_keyword][:, 0] - simulation_info[initial_keyword][:, 0]

        if reward_data["absolute_distance"]:
            return torch.abs(self.x_distance)
        else:
            return self.x_distance
        
    def _compute_final_torque_penalization_term_(self, simulation_info, reward_data):
        return self.torque_penalization_buffer
    
    def _compute_final_velocity_smoothness_term_(self, simulation_info, reward_data):
        return self.vel_smothness_buffer
    
    def _compute_final_limits_term_(self, simulation_info, reward_data):
        return self.limits_buffer

    def _compute_final_changed_actions_term_(self, simulation_info, reward_data):
        return self.changed_actions_buffer

    def _compute_final_y_distance_term_(self, simulation_info, reward_data):

        if self.y_distance is None:
            self.y_distance = simulation_info[root_keyword][:, 1] - simulation_info[initial_keyword][:, 1]

        if reward_data["absolute_distance"]:
            return torch.abs(self.y_distance)
        else:
            return self.y_distance

    def _compute_final_x_velocity_term_(self, simulation_info, reward_data):
        return self.accumulative_error_x_velocity

    def _compute_final_y_velocity_term_(self, simulation_info, reward_data):
        return self.accumulative_error_y_velocity

    def _compute_final_z_vel_term_(self, simulation_info, reward_data):
        return self.accumulative_error_z_speed

    def _compute_final_roll_pirch_term_(self, simulation_info, reward_data):
        return self.accumulative_error_roll_pitch

    def _compute_final_yaw_vel_term_(self, simulation_info, reward_data):
        return self.accumulative_error_yaw

    def _compute_final_slippery_term_(self, simulation_info, reward_data):
        return self.slippery_buffer

    def _compute_final_smoothness_term_(self, simulation_info, reward_data):
        return self.jerk_buffer

    def _compute_final_vel_cont_term_(self, simulation_info, reward_data):
        return self.accumulative_vel_cont_error

    def _compute_final_vibration_term_(self, simulation_info, reward_data):
        return self.accumulative_vibration_term

    def _compute_final_stability_term_(self, simulation_info, reward_data):
        rep = simulation_info[rep_keyword]
        rep_2 = rep * rep

        weights = reward_data["weights"]

        std_height_weight = weights["std_height"]
        mean_x_angle_weight = weights["mean_x_angle"]
        mean_y_angle_weight = weights["mean_y_angle"]
        mean_z_angle_weight = weights["mean_z_angle"]

        self.std_height = torch.sqrt(self.accumulative_height / rep - torch.square(self.accumulative_height_2) / rep_2)

        stability = std_height_weight * self.std_height
        stability += mean_x_angle_weight * self.acc[0] / rep
        stability += mean_y_angle_weight * self.acc[1] / rep
        stability += mean_z_angle_weight * self.acc[2] / rep

        if "distance" in weights:
            distance = self._compute_final_x_distance_term_(simulation_info, reward_data)
            stability *= weights["distance"] * distance

        return stability

    def _compute_final_speed_error_term_(self, simulation_info, reward_data):
        return self.accumulative_error_speed

    def _compute_final_low_penalization_contacts_term_(self, simulation_info, reward_data):
        weights = reward_data["weights"]
        max_clip = reward_data["max_clip"]

        correction_state_weight = weights["correction_state"]

        low_penalization_corrected = (self.low_penalization_contacts * correction_state_weight).clip(0, max_clip)

        if "distance" in weights:
            distance = self._compute_final_x_distance_term_(simulation_info, reward_data)
            low_penalization_corrected *= weights["distance"] * distance

        return low_penalization_corrected

    def _compute_final_high_penalization_contacts_term_(self, simulation_info, reward_data):
        weights = reward_data["weights"]
        max_clip = reward_data["max_clip"]
        high_penalization_corrected = 0.

        if max_clip != 0.0:
            correction_state_weight = weights["correction_state"]

            high_penalization_corrected = (self.high_penalization_contacts * correction_state_weight).clip(0, max_clip)

            if "distance" in weights:
                distance = self._compute_final_x_distance_term_(simulation_info, reward_data)
                high_penalization_corrected *= weights["distance"] * distance

        return high_penalization_corrected

    def _compute_final_height_error_term_(self, simulation_info, reward_data):
        max_clip = reward_data["max_clip"]
        return self.accumulative_height_error.clip(-max_clip, max_clip)

    def _compute_final_orthogonal_angle_error_term_(self, simulation_info, reward_data):
        return self.orthogonal_angle_error

    ###############################################################################################

    def _clean_buffer_x_distance_(self):
        self.x_distance = None
        self.x_distance_step = None

    def _clean_buffer_torque_penalization_(self):
        self.torque_penalization_buffer.fill_(0)

    def _clean_buffer_velocity_smoothness_(self):
        self.vel_smothness_buffer.fill_(0)

    def _clean_buffer_limits_(self):
        self.limits_buffer.fill_(0)

    def _clean_buffer_changed_actions_(self):
        self.changed_actions_buffer.fill_(0)

    def _clean_buffer_vel_cont_(self):
        self.accumulative_vel_cont_error.fill_(0)

    def _clean_buffer_slippery_(self):
        self.slippery_buffer.fill_(0)

    def _clean_buffer_z_vel_(self):
        self.accumulative_error_z_speed.fill_(0)

    def _clean_buffer_x_velocity_(self):
        self.accumulative_error_x_velocity.fill_(0)

    def _clean_buffer_y_velocity_(self):
        self.accumulative_error_y_velocity.fill_(0)

    def _clean_buffer_roll_pitch_(self):
        self.accumulative_error_roll_pitch.fill_(0)

    def _clean_buffer_yaw_vel_(self):
        self.accumulative_error_yaw.fill_(0)

    def _clean_buffer_smoothness_(self):
        self._prepare_buffer_smoothness_term_()

    def _clean_buffer_vibration_(self):
        self.accumulative_vibration_term.fill_(0)

    def _clean_buffer_y_distance_(self):
        self.y_distance = None
        self.y_distance_step = None

    def _clean_buffer_speed_error_(self):
        self.accumulative_error_speed.fill_(0)

    def _clean_buffer_stability_(self):
        self.accumulative_height.fill_(0)
        self.accumulative_height_2.fill_(0)
        self.acc.fill_(0)

    def _clean_buffer_low_penalization_contacts_(self):
        self.low_penalization_contacts.fill_(0)

    def _clean_buffer_high_penalization_contacts_(self):
        self.high_penalization_contacts.fill_(0)

    def _clean_buffer_height_error_(self):
        self.accumulative_height_error.fill_(0)

    def _clean_buffer_orthogonal_angle_error_(self):
        self.orthogonal_angle_error.fill_(0)


###############################################################################################


class Rewards(IndividualReward):
    def __init__(self, num_envs, device, rewards, gamma, steps, logger, discrete_rewards=False):
        super().__init__(num_envs, device)
        self.reward_terms = None
        self.gamma = gamma
        self.current_gama = 1.
        self.steps = steps
        self.iterations = 0
        self.previous_time = 0
        self.current_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rw_diff = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rw_nooise = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.discrete_rewards = discrete_rewards
        self.logger = logger
        self.change_rewards(rewards)
        self.ignore_rewards = [
            "ppo_penalization",
            "noise_ppo_penalization"
        ]
        self.constant_increase = 1.
        self.dic_rw = {}
        self.cst_ppo = 0.1

    def clear_constant_increase(self):
        self.constant_increase = 0.

    def increase_cst_ppo(self):
        self.cst_ppo += 0.1
    
    def increase_constant_by_step(self, step):
        self.constant_increase += step 

        if self.constant_increase >+ 1.:
            self.constant_increase = 1.
            return False
        else:
            return True

    def get_rewards(self):
        return self.reward_terms

    def change_rewards(self, rewards):
        self.reward_terms = rewards

    def prepare_buffers(self):
        for reward_name in self.reward_terms.keys():
            if reward_name in self.ignore_rewards:
                continue

            getattr(self, '_prepare_buffer_' + reward_name + '_term_')()

    def include_ppo_reward_penalization(self, penalization, noise, rewards, terrain_levels):
        if penalization is None:
            return rewards

        weight = self.reward_terms["ppo_penalization"]["weight"]
        weight_noise = self.reward_terms["noise_ppo_penalization"]["weight"]
        discount = self.reward_terms["ppo_penalization"]["discount_level"]

        if terrain_levels is not None:
            weight = weight / (1 + terrain_levels * discount)

        rw = weight * penalization / self.steps 
        self.rw_diff += rw
        # rw /= (1 + self.constant_increase/5)
        rewards += rw

        noise_rw = weight_noise * noise / self.steps * (self.cst_ppo * 10)
        self.rw_nooise += noise_rw
        # noise_rw /= (1 + self.constant_increase/4)
        # print(rw.mean(), noise_rw.mean())
        rewards += noise_rw

        self.current_reward += rewards

        return rewards

    def compute_rewards_in_state(self, simulation_info):
        rw = 0
        current_time = time.time()
        dt = 0.005 if not self.previous_time else current_time - self.previous_time

        simulation_info[dt_simulation_keyboard] = dt

        self.previous_time = time.time()

        for reward_name in self.reward_terms.keys():

            if reward_name in self.ignore_rewards:
                continue

            reward_data = None if not ("reward_data" in self.reward_terms[reward_name]) else \
                self.reward_terms[reward_name]['reward_data']
            weight = self.reward_terms[reward_name]['weight']
            cst = 1
            if "curriculum" in self.reward_terms[reward_name]:
                cst = self.constant_increase if self.reward_terms[reward_name]["curriculum"] else 1
                # print(reward_name, cst)

            ind_rw = getattr(self, '_compute_in_state_' + reward_name + '_term_')(simulation_info, reward_data)

            if weight == 0.0:
                continue

            # if self.iterations % 10 == 0:
            #     try:
            #         print(f'{reward_name}: {ind_rw.mean() * cst * weight}')
            #     except RuntimeError:
            #         pass

            rw += ind_rw * weight / self.steps * cst

            if not(reward_name is self.dic_rw):
                self.dic_rw[reward_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

            self.dic_rw[reward_name] += rw
            # if "high_penalization_contacts" == reward_name:
            #     print(ind_rw, rw)

        self.iterations += 1
        self.current_reward += rw * self.current_gama

        self.current_gama *= self.gamma
        return rw

    def compute_final_reward(self, simulation_info):
        reward = 0

        if self.discrete_rewards:
            return self.current_reward
        else:
            for reward_name in self.reward_terms.keys():
                if reward_name in self.ignore_rewards:
                    continue

                weight = self.reward_terms[reward_name]['weight']
                reward_data = None if not ("reward_data" in self.reward_terms[reward_name]) else \
                    self.reward_terms[reward_name]['reward_data']

                individual_reward = getattr(self, '_compute_final_' + reward_name + '_term_')(simulation_info,
                                                                                              reward_data)

                if weight == 0.0:
                    continue

                reward += individual_reward * weight

            return reward

    def clean_buffers(self):
        self.current_gama = 1.
        self.previous_time = 0
        self.current_reward.fill_(0)
        print(self.rw_diff.mean(), self.rw_nooise.mean())
        self.rw_nooise.fill_(0)
        self.rw_diff.fill_(0)

        for rw_n in self.dic_rw.keys():
            print(f"{rw_n}: {self.dic_rw[rw_n].mean()}", end="; ")
            self.dic_rw[rw_n].fill_(0)

        print()

        for reward_name in self.reward_terms.keys():
            if reward_name in self.ignore_rewards:
                continue

            getattr(self, '_clean_buffer_' + reward_name + '_')()

        self.iterations = 0

    def set_all_weights_zero(self, erase_points=None):
        for reward_name in self.reward_terms.keys():
            if reward_name in self.ignore_rewards:
                continue

            self.reward_terms[reward_name]['weight'] = 0.0

            if "curriculum" in self.reward_terms[reward_name]:
                self.reward_terms[reward_name]['curriculum'] = False

        if erase_points is not None:
            self.logger.new_interval_plot(erase_points)

    def save_weights(self, filename):
        self.logger.save_rewards(self.reward_terms, filename)



if __name__ == "__main__":
    reward_list = {
        "x_distance": {
            "weight": 1.,
            "reward_data": {
                "absolute_distance": True
            }
        },

        "y_distance": {
            "weight": -1.,
            "reward_data": {
                "absolute_distance": True
            }
        }
    }

    num_envs = 2
    device = 'cuda:0'

    root_sate = torch.zeros([num_envs, 3], dtype=torch.float, device=device, requires_grad=False)
    initial_root_sate = torch.zeros([num_envs, 3], dtype=torch.float, device=device, requires_grad=False)

    root_sate[:, 0].fill_(2.)
    root_sate[:, 1].fill_(-1.)

    simulation_info = {
        root_keyword: root_sate,
        initial_keyword: initial_root_sate
    }

    rewards_obj = Rewards(num_envs, device, reward_list, 0.99, 100)

    rewards_obj.prepare_buffers()
    rewards_obj.compute_rewards_in_state(simulation_info)
    final_rew = rewards_obj.compute_final_reward(simulation_info)

    print(final_rew)
