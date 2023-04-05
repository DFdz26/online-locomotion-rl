import torch


class Memory:
    class Step:
        def __init__(self):
            self.observations = None
            self.observation_expert = None
            self.actions = None
            self.critic_observations = None
            self.values = None
            self.rewards = None

            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_step_per_env, actions_shape, observation_shape, expert_observation_shape,
                 device='cpu'):
        self.num_step_per_env = num_step_per_env
        self.num_envs = num_envs

        self.step = 0

        self.device = device
        self.actions_shape = actions_shape

        # Simulation memory
        self.observations = torch.zeros(num_step_per_env, num_envs, observation_shape, device=self.device,
                                        requires_grad=False)
        self.observation_expert = torch.zeros(num_step_per_env, num_envs, expert_observation_shape, device=self.device,
                                        requires_grad=False)
        self.rewards = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False)
        self.actions = torch.zeros(num_step_per_env, num_envs, actions_shape, device=self.device, requires_grad=False)
        self.dones = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False).byte()

        # Actor critic memory
        self.actions_log_prob = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False)
        self.values = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False)
        self.returns = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False)
        self.advantages = torch.zeros(num_step_per_env, num_envs, 1, device=self.device, requires_grad=False)
        self.mu = torch.zeros(num_step_per_env, num_envs, actions_shape, device=self.device, requires_grad=False)
        self.sigma = torch.zeros(num_step_per_env, num_envs, actions_shape, device=self.device, requires_grad=False)

    def add_steps_into_memory(self, step: Step):
        if self.step >= self.num_step_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(step.observations)
        self.observation_expert[self.step].copy_(step.observation_expert)
        self.dones[self.step].copy_(step.dones.view(-1, 1))
        self.actions[self.step].copy_(step.actions)
        self.rewards[self.step].copy_(step.rewards.view(-1, 1))
        self.values[self.step].copy_(step.values)
        self.actions_log_prob[self.step].copy_(step.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(step.action_mean)
        self.sigma[self.step].copy_(step.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    # “reward-to-go policy gradient” Compute first the last returns, it's easier and from that compute A.
    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_step_per_env)):
            if step == self.num_step_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_step_per_env
        mini_batch_size = batch_size // num_mini_batches

        # Randomize the samples optimizes the learning process
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        expert_observations = self.observation_expert.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                expert_obs = expert_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, expert_obs, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, \
                    returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
