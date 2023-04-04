import torch


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
    def __init__(self, PPO, PIBB, Rw) -> None:
        self.PPO = PPO
        self.PIBB = PIBB
        self.iteration = 0
        self.start_act_PPO = 300
        self.it = 0
        self.gamma = 0.1
        self.gamm_it = 0
        self.restore_noise = True
        self.rewards = Rw

        self.max_fitness = {
            "record": -999,
            "iteration": 0
        }

        self.history_PPO_fitness = {
            "record": -999,
            "minimum": 999,
            "iteration_max": 0,
            "iteration_min": 0,
            "mean": 0,
            "counter": 0
        }
        self.accum_fitness = 0.

    def _prepare_MLP_(self):
        if self.restore_noise:
            self.restore_noise = False

            self.PIBB.noise_arr.fill_(0)
            self.PIBB.policy.mn.apply_noise_tensor(self.PIBB.noise_arr)
        pass

    def update(self, policy, rewards):
        
        if self.iteration > self.start_act_PPO:
            self.gamm_it += 1
            # self.PPO.memory.clear()
            # return None
            self.iteration += 1
            return self.PPO.update(policy.get_MLP(), rewards * 1000000)
        else:

            self.PIBB.update(policy.get_CPG_RBFN(), rewards*0.5 if self.iteration > self.start_act_PPO else rewards)
            self.iteration += 1

            # if self.iteration > self.start_act_PPO:
            #     rewards = self.rewards.get_rewards
            #     rewards['x_distance']['weight'] *= 1.5
            #     rewards['x_velocity']['weight'] *= 1.5
            return None

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

        if rep > (self.start_act_PPO + 1):
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
        print(f"Rep: {rep}, gamma: {self.gamma}, gamma_it: {self.gamm_it}")
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
        
        if rep > (self.start_act_PPO + 1):
            print(f"Max PPO fitness: {self.history_PPO_fitness['record']} at iteration: {self.history_PPO_fitness['iteration_max']}")
            print(f"Min PPO fitness: {self.history_PPO_fitness['minimum']} at iteration: {self.history_PPO_fitness['iteration_min']}")
            print(f"Mean PPO fitness: {self.history_PPO_fitness['mean']/(self.history_PPO_fitness['counter']) * 1000}")

        print("=============================")

    def prepare_training(self, num_envs, steps_per_env, observations, actions, policy):
        self.PPO.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_MLP())
        self.PIBB.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_CPG_RBFN())

    def post_step_simulation(self, obs, actions, reward, dones, info, closed_simulation):
        if not closed_simulation:
            if self.iteration > self.start_act_PPO:
                self.PPO.post_step_simulation(obs, actions, reward * self.gamma * 1000, dones, info, closed_simulation)
            else:
                self.PIBB.post_step_simulation(obs, actions, reward, dones, info, closed_simulation)

    def last_step(self, obs):
        self.PIBB.last_step(obs)

        if self.iteration > self.start_act_PPO:
            self.PPO.last_step(obs)

    def get_noise(self):
        return self.PIBB.get_noise()
    
    def act(self, observation):
        if self.iteration > self.start_act_PPO:
            if not self.change:
                self.PIBB.noise_arr.fill_(0)
                self.PIBB.policy.mn.apply_noise_tensor(self.PIBB.noise_arr)
                self.change = True
            
            actions_PPO = self.PPO.act(observation, actions_mult=1.)
            actions_CPG = self.PIBB.act(observation) * 1.

            for i in range(len(actions_PPO)):
                
                new = [ (actions_PPO[i] - torch.min(actions_PPO[i])) / (torch.max(actions_PPO[i]) - torch.min(actions_PPO[i])) - 0.5] * 4
                actions_PPO[i] = new[0]

            # if self.it == 0:
            #     print(f'actions PPO: {actions_PPO[0]}')
            #     print(f'actions_CPG: {actions_CPG[0]}')

            # self.it += 1
            # self.it %= 50

            # actions = (1-self.gamma) * actions_CPG + self.gamma * actions_PPO
            actions = actions_CPG + 0.1 * actions_PPO

            if self.gamm_it > 2000:
                if self.gamma < 0.5:
                    self.gamma += 0.1
                self.gamm_it = 0

            return actions
        else:

            actions = self.PIBB.act(observation)
            self.change = False
        
            # if self.iteration > 80:
            #     actions[:, :2] = 0.
            return actions 

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