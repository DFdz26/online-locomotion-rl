"""
Class: ActorCritic
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 26 February 2023

Actor Critic
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


accepted_kwargs = ["debug_mess"]


class NNCreatorArgs:
    inputs = []
    outputs = []
    hidden_dim = []
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid or none


class ActorCritic(nn.Module):

    def __init__(self, actorArgs, criticArgs, actor_std_noise, **kwargs):

        super().__init__()
        self.__default_values_kwargs__()

        if kwargs:
            self.__prepare_kwargs__(kwargs)

        self.__actor_building__(actorArgs)
        self.__critic_building__(criticArgs)

        if self.debug_mess:
            print(f"Actor MLP: {self.actor_body}")
            print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(actor_std_noise * torch.ones(actorArgs.outputs[0]))
        self.distribution = None

        # For optimizing the process
        Normal.set_default_validate_args = False

    def forward(self):
        pass

    def reset(self):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        selected_action = self.actor_NN(observations)
        self.distribution = Normal(selected_action, self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations):
        return self.critic_NN(critic_observations)

    def __default_values_kwargs__(self):
        self.debug_mess = False

    def __prepare_kwargs__(self, kwargs):
        not_accepted, accepted = check_if_args_in_accepted(kwargs)

        if len(not_accepted):
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + not_accepted)

        if "debug_mess" in accepted:
            self.debug_mess = accepted["debug_mess"]

    def __actor_building__(self, actorArgs):
        if self.debug_mess:
            print("Starting to build the Actor")

        layers = self.__generic_MLP_building__(actorArgs)

        if self.debug_mess:
            print("Creating the Critic")

        self.actor_NN = nn.Sequential(*layers)

    def __critic_building__(self, criticArgs):
        if self.debug_mess:
            print("Starting to build the Critic")

        layers = self.__generic_MLP_building__(criticArgs)

        if self.debug_mess:
            print("Creating the Critic")

        self.critic_NN = nn.Sequential(*layers)

    @staticmethod
    def __generic_MLP_building__(args):
        activation = get_activation(args.activation)

        layers = []
        size_input = args.inputs[0]

        for h in range(len(args.hidden_dim)):
            layers.append(nn.Linear(size_input, args.critic_hidden_dims[h]))

            if not(activation is None):
                layers.append(activation)

            size_input = args.critic_hidden_dims[h]

        layers.append(nn.Linear(size_input, args.outputs[0]))

        return layers


def check_if_args_in_accepted(args):
    not_accepted = ""
    accepted = {}

    for k in args.keys():
        if not (k in accepted_kwargs):
            not_accepted += k + ", "
        else:
            accepted[k] = args[k]

    if len(not_accepted):
        not_accepted = not_accepted[:2]

    return not_accepted, accepted


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "none":
        return None
    else:
        raise Exception(f"invalid activation function: {act_name}")