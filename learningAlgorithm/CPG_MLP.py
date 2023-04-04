import torch
import pickle


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

    def update(self, policy, rewards):
        
        return self.curricula.update_algorithm(policy, rewards)

    def print_info(self, rw, rep, total_time, rollout_time, loss):
        if loss is None:
            loss = {
                'mean_surrogate_loss': 0,
                'mean_value_loss': 0,
                'mean_loss': 0,
                'entropy': 0,
                'lr': 0,
                'kl_mean': 0,
            }

        mean_fitness = float(torch.mean(rw)) 
        self.accum_fitness += mean_fitness

        if mean_fitness > self.max_fitness["record"]:
            self.max_fitness["record"] = mean_fitness
            self.max_fitness['iteration'] = rep

        if self.curricula.learning.PPO_learning_activated:
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
        print(f"Rep: {rep}, gamma: {self.curricula.learning.gamma}, gamma_it: {self.curricula.learning.count_increase_gamma}")
        print(f"Mean fitness: {mean_fitness}")
        print(f"Max fitness: {self.max_fitness['record']} at iteration: {self.max_fitness['iteration']}")
        print(f"Mean accu fitness: {self.accum_fitness/(rep + 1) * 1000}")
        print(f"PPO::: Value loss: {loss['mean_value_loss']}", end="\t")
        print(f"Surrogate loss: {loss['mean_surrogate_loss']}", end="\t")
        print(f"Learning rate: {loss['lr']}", end="\t")
        print(f"KL mean: {loss['kl_mean']}", end="\t")
        print(f"Loss mean: {loss['mean_loss']}", end="\t")
        print(f"Entropy: {loss['entropy']}")
        print(f"PIBB::: variance: {self.PIBB.variance}")
        print(f"Total time (s): {total_time}")
        print(f"Rollout time (s): {rollout_time}")
        
        if self.curricula.learning.PPO_learning_activated:
            print(f"Max PPO fitness: {self.history_PPO_fitness['record']} at iteration: {self.history_PPO_fitness['iteration_max']}")
            print(f"Min PPO fitness: {self.history_PPO_fitness['minimum']} at iteration: {self.history_PPO_fitness['iteration_min']}")
            print(f"Mean PPO fitness: {self.history_PPO_fitness['mean']/(self.history_PPO_fitness['counter']) * 1000}")

        print("=============================")

    def prepare_training(self, num_envs, steps_per_env, observations, actions, policy):
        self.PPO.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_MLP())
        self.PIBB.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_CPG_RBFN())

    def post_step_simulation(self, obs, actions, reward, dones, info, closed_simulation):
        if not closed_simulation:
            self.curricula.post_step_simulation(obs, actions, reward, dones, info, self.PPO, self.PIBB)

    def last_step(self, obs):
        self.curricula.last_step(obs, self.PPO, self.PIBB)

    def get_noise(self):
        return self.PIBB.get_noise()
    
    def act(self, observation):
        if self.test:
            PPO_actions = self.PPO.act(observation)
            PIBB_actions = self.PIBB.act(observation)

            return self.learnt_weight["PIBB"] * PIBB_actions + self.learnt_weight["PPO"] * PPO_actions
        else:
            return self.curricula.act_curriculum(observation, self.PPO, self.PIBB)


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
