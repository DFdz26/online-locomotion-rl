import sys
import math
import torch


class PIBB(object):
    def __init__(self, _rollouts, _h, _decay_h, _noise_len, decay, variance, device="cpu", boost_noise=1):
        self.device = device
        self.rollouts = _rollouts
        self.h = _h
        self.decay_h = _decay_h
        self.variance = variance
        self.policy = None  # Set once the training starts

        self.decay = decay
        self.variance_ex = variance

        self.p = torch.zeros(_rollouts, dtype=torch.float32, device=self.device, requires_grad=False)
        self.s_norm = torch.ones(_rollouts, dtype=torch.float32, device=self.device, requires_grad=False)
        self.cost_weighted_noise = None
        self.noise_arr = None
        self.output_policy = None
        self.previous_noise = None
        self.previous_reward = None

        self.create_noise_cost_weighted_noise(_noise_len, boost_noise=boost_noise)

    def get_p(self):
        return self.p

    def create_noise_cost_weighted_noise(self, _noise_len, boost_noise=1., new_variance=None, new_decay=None):
        if not(new_decay is None):
            self.decay = new_decay

        if not(new_variance is None):
            self.variance = new_variance
            self.variance_ex = new_variance

        self.cost_weighted_noise = torch.zeros(_noise_len, dtype=torch.float32, device=self.device, requires_grad=False)

        self.noise_arr = torch.zeros([self.rollouts, _noise_len], dtype=torch.float32, device=self.device,
                                     requires_grad=False).normal_(mean=0., std=math.sqrt(self.variance) * boost_noise)

    def get_h(self):
        return self.h
    
    def get_rbf_activations(self):
        return self.policy.get_last_rbfn_activations()

    def load_policy_weights(self, weights):
        self.policy.load_weights(weights)

    def get_policy_weights(self):
        return self.policy.get_weights()

    def genere_noise_arr(self, variance=None):
        if variance is None:
            variance = self.variance

        self.noise_arr.normal_(0., std=math.sqrt(variance))

    def get_noise(self):
        return self.noise_arr

    def get_hyper_parameters(self):
        hyp = {
            "device": self.device,
            "rollouts": self.rollouts,
            "h": self.h
        }

        if not (self.decay is None):
            hyp["decay"] = self.decay

        if not (self.variance_ex is None):
            hyp["variance"] = self.variance_ex

        return hyp

    def print_info(self, rw, rep, total_time, rollout_time, loss):
        max_fitness = float(torch.max(rw))
        min_fitness = float(torch.min(rw))
        mean_fitness = float(torch.mean(rw))

        print("=============================")
        print(f"Rep: {rep}")
        print(f"Max: {max_fitness}", end="\t")
        print(f"Min: {min_fitness}", end="\t")
        print(f"Avg: {mean_fitness}")
        print(f"variance: {self.variance}")
        print(f"Total time (s): {total_time}")
        print(f"Rollout time (s): {rollout_time}")
        print("=============================")

    def _compute_s_norm_(self, fitness_arr, max_fitness, min_fitness):

        for k in range(self.rollouts):
            self.s_norm[k] = math.exp(self.h * ((fitness_arr[k] - min_fitness) / (max_fitness - min_fitness)))

    def step(self, fitness_arr, _parameter_arr):
        parameter_arr = _parameter_arr.detach().clone().to(self.device).flatten()

        # Calculate fitness min, max, and avg.
        max_fitness = float(torch.max(fitness_arr))
        min_fitness = float(torch.min(fitness_arr))

        # Compute trajectory cost/fitness
        if max_fitness != min_fitness:
            self._compute_s_norm_(fitness_arr, max_fitness, min_fitness)
        else:
            self.s_norm.fill_(1)

        total_s = float(torch.sum(self.s_norm))

        # Compute probability for each roll-out
        for k in range(self.rollouts):
            p = self.s_norm[k] / total_s
            # Cost-weighted averaging
            self.cost_weighted_noise = p * self.noise_arr[k]
            # noise_arr[k]    = [x * self.p[k] for x in noise_arr[k]]
            # Update policy parameters
            # parameter_arr   = list(map(add, parameter_arr, self.cost_weighted_noise))
            # print(parameter_arr)
            # print(self.cost_weighted_noise)
            parameter_arr += self.cost_weighted_noise

        return torch.reshape(parameter_arr, _parameter_arr.shape)

    def update(self, policy, rewards):
        policy.modify_weights(self.step(rewards, policy.get_weights()[0]))
        self.post_step()
        policy.mn.apply_noise_tensor(self.get_noise())
        self.policy = policy

    def post_step(self):
        # Decay h (1/λ) / variance as learning progress
        self.h = self.decay_h * (1 / self.h)
        self.h = (1 / self.h)

        self.variance *= self.decay
        self.genere_noise_arr()

    def prepare_training(self, agents, steps_per_iteration, num_observation, expert_obs, num_actions, policy):
        policy.mn.apply_noise_tensor(self.get_noise())
        self.policy = policy

    @staticmethod
    def post_step_simulation(obs, exp_obs, actions, reward, dones, info, closed_simulation):
        pass

    def last_step(self, obs, exp_obs, reset=True):
        if reset:
            self.policy.reset()
        return self.output_policy

    def act(self, obs, obs_exp, modification_amplitude=0.0, phase_shift=1.0, dt=None, action_mult=1.):
        self.output_policy = self.policy.forward(amplitude_change=modification_amplitude, frequency_change=phase_shift, dt=dt, output_mult=action_mult)

        return self.output_policy

    def change_max_frequency_cpg(self, new_max, dt=None):
        self.policy.change_maximum_speed_cpg(new_max, dt)
