import torch

rep_keyword = 'rep'
root_keyword = 'root_state'
base_lin_vel_keyboard = 'base_lin_vel'
base_ang_vel_keyboard = 'base_ang_vel'
previous_position_keyword = 'previous_position'
initial_keyword = 'initial_state'
projected_gravity_keyword = 'projected_gravity'
contact_forces_gravity_keyword = 'contact_forces'
termination_contact_indices_keyword = 'termination_contact_indices'
penalization_contact_indices_keyword = 'penalization_contact_indices'
goal_height_keyword = 'goal_height'


class IndividualReward:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.x_distance = None
        self.y_distance = None

    ###############################################################################################

    """
    Preparation of the buffers
    """

    def _prepare_buffer_x_distance_term_(self):
        pass

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

    def _prepare_buffer_height_error_term_(self):
        self.accumulative_height_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                     requires_grad=False)

    ###############################################################################################
    """
    Compute the needed calculations in every step
    """

    def _compute_in_state_x_distance_term_(self, simulation_info, reward_data):
        previous_position = simulation_info[previous_position_keyword][:, 0]
        actual_position = simulation_info[root_keyword][:, 0]

        self.x_distance_step = actual_position - previous_position
        return self.x_distance_step

    def _compute_in_state_y_distance_term_(self, simulation_info, reward_data):
        previous_position = simulation_info[previous_position_keyword][:, 1]
        actual_position = simulation_info[root_keyword][:, 1]

        self.y_distance_step = actual_position - previous_position
        return self.y_distance_step

    def _compute_in_state_speed_error_term_(self, simulation_info, reward_data):
        def error_in_speed_int(actual, desired, compute, division_):
            if compute:
                error = actual - desired
                return torch.exp(-torch.torch.square(error)/division_)
            else:
                return 0

        base_vel = simulation_info[base_lin_vel_keyboard]
        speed_in_x_rw = reward_data['speed_in_x']
        speed_in_y_rw = reward_data['speed_in_y']
        speed_in_z_rw = reward_data['speed_in_z']
        division = reward_data['division']

        error_z = error_in_speed_int(speed_in_z_rw, base_vel[:, 2], not(None is speed_in_z_rw), division)
        error_x = error_in_speed_int(speed_in_x_rw, base_vel[:, 0], not(None is speed_in_x_rw), division)
        error_y = error_in_speed_int(speed_in_y_rw, base_vel[:, 1], not(None is speed_in_y_rw), division)

        in_state = error_z + error_x + error_y
        self.accumulative_error_speed += in_state

        return in_state

    def _compute_in_state_stability_term_(self, simulation_info, reward_data):
        projected_gravity = simulation_info[projected_gravity_keyword]
        root_state = simulation_info[root_keyword]

        for i in range(3):
            self.acc[i] += projected_gravity[:, i].detach()

        self.accumulative_height += root_state[:, 2]
        self.accumulative_height_2 += torch.square(root_state[:, 2])

    def _compute_in_state_low_penalization_contacts_term_(self, simulation_info, reward_data):
        contact_forces = simulation_info[contact_forces_gravity_keyword]
        penalization_contact_indices = simulation_info[penalization_contact_indices_keyword]

        in_state = torch.any(torch.norm(contact_forces[:, penalization_contact_indices, :], dim=-1) > 1., dim=1)
        self.low_penalization_contacts += in_state

        return in_state

    def _compute_in_state_high_penalization_contacts_term_(self, simulation_info, reward_data):
        contact_forces = simulation_info[contact_forces_gravity_keyword]
        termination_contact_indices = simulation_info[termination_contact_indices_keyword]

        self.high_penalization_contacts += torch.any(
            torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1) > 1., dim=1)

    def _compute_in_state_height_error_term_(self, simulation_info, reward_data):
        z_state = simulation_info[root_keyword][:, 2]
        goal_height = simulation_info[goal_height_keyword]

        self.accumulative_height_error += torch.sqrt(torch.abs(z_state - goal_height)) / 100

    ###############################################################################################

    def _compute_final_x_distance_term_(self, simulation_info, reward_data):

        if self.x_distance is None:
            self.x_distance = simulation_info[root_keyword][:, 0] - simulation_info[initial_keyword][:, 0]

        if reward_data["absolute_distance"]:
            return torch.abs(self.x_distance)
        else:
            return self.x_distance

    def _compute_final_y_distance_term_(self, simulation_info, reward_data):

        if self.y_distance is None:
            self.y_distance = simulation_info[root_keyword][:, 1] - simulation_info[initial_keyword][:, 1]

        if reward_data["absolute_distance"]:
            return torch.abs(self.y_distance)
        else:
            return self.y_distance

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

    ###############################################################################################

    def _clean_buffer_x_distance_(self):
        self.x_distance = None
        self.x_distance_step = None

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


###############################################################################################

# TODO: test with jit script, reduces the computational time
# @torch.jit.script
class Rewards(IndividualReward):
    def __init__(self, num_envs, device, rewards, gamma):
        super().__init__(num_envs, device)
        self.reward_terms = None
        self.gamma = gamma
        self.actual_gama = gamma
        self.change_rewards(rewards)

    def change_rewards(self, rewards):
        self.reward_terms = rewards

    def prepare_buffers(self):
        for reward_name in self.reward_terms.keys():
            getattr(self, '_prepare_buffer_' + reward_name + '_term_')()

    def compute_rewards_in_state(self, simulation_info):
        for reward_name in self.reward_terms.keys():
            getattr(self, '_compute_in_state_' + reward_name + '_term_')(simulation_info)

        self.actual_gama *= self.actual_gama

    def compute_final_reward(self, simulation_info):
        reward = 0
        for reward_name in self.reward_terms.keys():
            weight = self.reward_terms[reward_name]['weight']
            reward_data = None if not ("reward_data" in self.reward_terms[reward_name]) else \
                self.reward_terms[reward_name]['reward_data']

            individual_reward = getattr(self, '_compute_final_' + reward_name + '_term_')(simulation_info, reward_data)

            if weight == 0.0:
                continue

            reward += individual_reward * weight

        return reward

    def clean_buffers(self):
        self.actual_gama = self.gamma

        for reward_name in self.reward_terms.keys():
            getattr(self, '_clean_buffer_' + reward_name + '_')()


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

    rewards_obj = Rewards(num_envs, device, reward_list)

    rewards_obj.prepare_buffers()
    rewards_obj.compute_rewards_in_state(simulation_info)
    final_rew = rewards_obj.compute_final_reward(simulation_info)

    print(final_rew)
