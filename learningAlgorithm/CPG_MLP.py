"""
Class: PPO_PIBB
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 15 February 2023

Fusion PPO and PIBB algorithms
"""

import torch
import pickle


# from PPO.PPO import PPO as PPO_class
# from PIBB.PIBB import PIBB as PIBB_class


class MLP_CPG:
    def __init__(self, MLP, CPG) -> None:
        self.MLP = MLP
        self.CPG = CPG

    def get_weights(self):
        return [self.CPG.get_weights(), self.MLP.get_weights()]

    def get_MLP(self):
        return self.MLP

    def get_CPG_RBFN(self):
        return self.CPG

    def load_weights(self, file, load_CPG=True, load_MLP=True):
        if load_CPG:
            self.CPG.load_weights()

        if load_MLP:
            self.MLP.load_weights()


class PPO_PIBB:
    def __init__(self, PPO, PIBB, Curricula, learnt_weights=None, test=False) -> None:
        self.PPO = PPO
        self.PIBB = PIBB
        self.it = 0
        self.curricula = Curricula
        self.test = True
        self.learnt_weight = {}
        self.PPO_info = None
        self.gamma_PPO = 0.
        self.rep = 0

        if learnt_weights is None:
            self.test = False

        self.learnt_weight = learnt_weights

        self.max_fitness = {
            "record": -999.,
            "iteration": 0
        }

        self.history_PPO_fitness = {
            "record": -999.,
            "minimum": 999.,
            "iteration_max": 0,
            "iteration_min": 0,
            "mean": 0.,
            "counter": 0
        }
        self.accum_fitness = 0.

    def read_data_point(self, filename, logger, policy, post=True, recover_MLP=True, recover_CPG=True, test=False):
        weights, noise, iteration, _, index = logger.recover_data(filename, post)

        MLP_weights = weights[0]
        CPG_weights = weights[1]
        combined = weights[2]

        if recover_MLP:
            self.PPO.load_policy_weights(MLP_weights)

        if recover_CPG:
            self.PIBB.policy = policy.CPG
            self.PIBB.load_policy_weights(CPG_weights)

        self.learnt_weight = combined
        self.test = test

        if test:
            self.PPO.test_mode()

        return noise, iteration, index

    def get_info_algorithm(self, get_PPO=True, get_PIBB=True):
        PPO_cfg = None
        PIBB_cfg = None

        if get_PPO:
            PPO_cfg = self.PPO.cfg

        if get_PIBB:
            PIBB_cfg = self.PIBB.get_hyper_parameters()

        return [PPO_cfg, PIBB_cfg]

    def get_weights_policy(self):
        return [self.PPO.get_policy_weights(), self.PIBB.get_policy_weights(), self.curricula.get_weights_NN()]

    def update(self, policy, rewards):

        return self.curricula.update_algorithm(policy, rewards, self.PPO, self.PIBB)
    
    def _get_curriculum_levels(self):
        if self.curricula is None:
            return 0, 0, 0
        
        self.curricula.get_levels_curriculum()
    
    def get_last_PPO_info_for_logger(self):
        if self.PPO_info is None:
            return None
        
        terrain, rand = self.curricula.get_levels_curriculum()
        
        ppo_logger = {
            "gamma": self.gamma_PPO,
            "lr": self.PPO_info["lr"],
            "surrogate_loss": self.PPO_info["mean_surrogate_loss"],
            "value_loss": self.PPO_info["mean_value_loss"],
            "entropy": float(self.PPO_info["entropy"]),
            "student_loss": self.PPO_info["student_loss"],
            "terrain_curriculum": terrain,
            "loss_mean": self.PPO_info["mean_loss"],
            "time_steps_PPO": self.rep,
        }

        return ppo_logger

    def print_info(self, rw, rep, total_time, rollout_time, loss, length_ep, loss_AC_supervised):
        self.PPO_info = loss
        self.rep = rep

        if loss is None:
            loss = {
                'mean_surrogate_loss': 0,
                'mean_value_loss': 0,
                'mean_loss': 0,
                'entropy': 0,
                'lr': 0,
                'kl_mean': 0,
                'student_loss': 0,
                "mean_difference_cpg":0
            }

        mean_fitness = float(torch.mean(rw))
        self.accum_fitness += mean_fitness

        if mean_fitness > self.max_fitness["record"]:
            self.max_fitness["record"] = mean_fitness
            self.max_fitness['iteration'] = rep

        if self.curricula.algorithm_curriculum.PPO_learning_activated:
            if mean_fitness > self.history_PPO_fitness["record"]:
                self.history_PPO_fitness["record"] = mean_fitness
                self.history_PPO_fitness["iteration_max"] = rep

            if mean_fitness < self.history_PPO_fitness["minimum"]:
                self.history_PPO_fitness["minimum"] = mean_fitness
                self.history_PPO_fitness["iteration_min"] = rep

            self.history_PPO_fitness["mean"] += mean_fitness
            self.history_PPO_fitness["counter"] += 1

        mean_fitness *= 1000

        print("=============================")
        print(f"Rep: {rep}, gamma: {self.curricula.algorithm_curriculum.gamma}, "
              f"gamma_it: {self.curricula.algorithm_curriculum.count_increase_gamma}")
        print(f"Mean fitness: {mean_fitness}")
        print(f"Max fitness: {self.max_fitness['record']} at iteration: {self.max_fitness['iteration']}")
        print(f"Mean accu fitness: {self.accum_fitness / (rep + 1) * 1000}")
        print(f"PPO::: Value loss: {loss['mean_value_loss']}", end="\t")
        print(f"Surrogate loss: {loss['mean_surrogate_loss']}", end="\t")
        print(f"Learning rate: {loss['lr']}", end="\t")
        print(f"KL mean: {loss['kl_mean']}", end="\t")
        print(f"Loss mean: {loss['mean_loss']}", end="\t")
        print(f"Entropy: {loss['entropy']}")
        print(f"Student loss: {loss['student_loss']}")
        print(f"cpg loss: {loss['mean_difference_cpg']}")
        print(f"PIBB::: variance: {self.PIBB.variance}")
        print(f"Total time (s): {total_time}")
        print(f"Rollout time (s): {rollout_time}")
        print(f"Length episode : {length_ep}")

        if not (loss_AC_supervised is None):
            print(f"Loss AC as student: {loss_AC_supervised}")

        if self.curricula.algorithm_curriculum.PPO_learning_activated:
            print(f"Max PPO fitness: {self.history_PPO_fitness['record']} at iteration: "
                  f"{self.history_PPO_fitness['iteration_max']}")
            print(f"Min PPO fitness: {self.history_PPO_fitness['minimum']} at iteration: "
                  f"{self.history_PPO_fitness['iteration_min']}")
            print(
                f"Mean PPO fitness: {self.history_PPO_fitness['mean'] / (self.history_PPO_fitness['counter']) * 1000}")

        print("=============================")

    def prepare_training(self, num_envs, steps_per_env, observations, expert_obs, actions, policy):
        self.PPO.prepare_training(num_envs, steps_per_env, observations, expert_obs, actions, policy.get_MLP())
        self.PIBB.prepare_training(num_envs, steps_per_env, observations, expert_obs, actions, policy.get_CPG_RBFN())

    def post_step_simulation(self, obs, exp_obs, actions, reward, dones, info, closed_simulation):
        if not closed_simulation:
            self.curricula.post_step_simulation(obs, exp_obs, actions, reward, dones, info, self.PPO, self.PIBB)

    def last_step(self, obs, exp_obs):
        self.curricula.last_step(obs, exp_obs, self.PPO, self.PIBB)

    def get_noise(self):
        return self.PIBB.get_noise()

    def act(self, observation, expert_obs, history_obs):
        if self.test:
            PPO_actions = self.PPO.act(observation, expert_obs, history_obs)
            PIBB_actions = self.PIBB.act(observation, expert_obs)
            self.gamma_PPO = 0.5

            return 0.5 * PIBB_actions + self.gamma_PPO * PPO_actions
            # return PIBB_actions 
            # return self.learnt_weight["PPO"]* PPO_actions 
        else:
            action, rw_ppo_diff_cpg = self.curricula.act_curriculum(observation, expert_obs, history_obs, self.PPO, self.PIBB)
            self.gamma_PPO = self.curricula.algorithm_curriculum.gamma

            return action, rw_ppo_diff_cpg


if __name__ == "__main__":
    from PPO.ActorCritic import ActorCritic
    from PPO.ActorCritic import NNCreatorArgs
    from PPO.PPO import PPO

    from PIBB.PIBB import PIBB

    actorArgs = NNCreatorArgs()
    actorArgs.inputs = [42]
    actorArgs.hidden_dim = [32, 12]
    actorArgs.outputs = [5]

    criticArgs = NNCreatorArgs()
    criticArgs.inputs = [42]
    criticArgs.hidden_dim = [32, 12]
    criticArgs.outputs = [1]

    actor_std_noise = 1.

    actorCritic = ActorCritic(actorArgs, criticArgs, actor_std_noise, debug_mess=True)
    ppo = PPO(actorCritic, device="cuda:0", verbose=True)

    print('creating memory')

    ppo.init_memory(15, 100, 42, 12)
    print('Created memory')

    action = ppo.act(torch.zeros(42, dtype=torch.float, device="cuda:0", requires_grad=False), actions_mult=0.5)
    print(action)
