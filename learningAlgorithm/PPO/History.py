import torch


class History:
    def __init__(self, num_env, num_observations, observation_shape=None, device=None):
        self.observation_shape = observation_shape
        self.num_observations = num_observations
        self.device = device if device is not None else torch.device('cpu')
        self.num_env = num_env
        self.history_shape = None
        self.history = None

        if self.observation_shape is not None:
            self.set_up_history(observation_shape)

    def set_up_history(self, observations_shape):
        self.observation_shape = observations_shape
        self.history_shape = self.observation_shape * self.num_observations
        self.history = torch.zeros(self.num_env, self.history_shape, device=self.device)

    def store_history(self, observations):
        self.history = torch.cat([self.history[:, self.observation_shape:], observations], dim=1)
        return self.history

    def reset_history(self):
        self.history.zero_()

    def reset_specific_history(self, env_id):
        self.history[env_id].zero_()
