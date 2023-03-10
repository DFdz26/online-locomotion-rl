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

class PPO_PIBB:
    def __init__(self, PPO, PIBB) -> None:
        self.PPO = PPO
        self.PIBB = PIBB

    def update(self, policy, rewards):
        self.PIBB.update(policy.get_CPG_RBFN(), rewards)
        return self.PPO.update(policy.get_MLP(), rewards)

    def print_info(self, rw, rep, total_time, rollout_time, loss):
        mean_fitness = float(torch.mean(rw))

        print("=============================")
        print(f"Rep: {rep}")
        print(f"Mean fitness: {mean_fitness}")
        print(f"PPO::: Value loss: {loss['mean_value_loss']}", end="\t")
        print(f"Surrogate loss: {loss['mean_surrogate_loss']}")
        print(f"PIBB::: variance: {self.PIBB.variance}")
        print(f"Total time (s): {total_time}")
        print(f"Rollout time (s): {rollout_time}")
        print("=============================")

    def prepare_training(self, num_envs, steps_per_env, observations, actions, policy):
        self.PPO.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_MLP())
        self.PIBB.prepare_training(num_envs, steps_per_env, observations, actions, policy.get_CPG_RBFN())

    def post_step_simulation(self, obs, actions, reward, dones, info, closed_simulation):
        if not closed_simulation:
            self.PPO.post_step_simulation(obs, actions, reward, dones, info, closed_simulation)
            self.PIBB.post_step_simulation(obs, actions, reward, dones, info, closed_simulation)

    def last_step(self, obs):
        self.PIBB.last_step(obs)
        self.PPO.last_step(obs)

    def get_noise(self):
        return self.PIBB.get_noise()
    
    def act(self, observation):
        actions_PPO = self.PPO.act(observation, actions_mult=1.)
        actions_CPG = self.PIBB.act(observation) * 1.

        print(f'actions PPO: {actions_PPO[0]}')
        print(f'actions_CPG PPO: {actions_CPG[0]}')

        return actions_CPG + actions_PPO


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