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

accepted_kwargs = ["debug_mess", "test", "scale_max", "scale_min"]


class NNCreatorArgs:
    inputs = []
    outputs = []
    hidden_dim = []
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid or none


class ActorCritic(nn.Module):

    def __init__(self, actorArgs, criticArgs, actor_std_noise, expertArgs, studentArgs, **kwargs):

        super().__init__()
        self.__default_values_kwargs__()
        self.actor_std_noise = actor_std_noise
        self.scale_output = False

        if kwargs:
            self.__prepare_kwargs__(kwargs)

        if not self.test:
            self.__actor_building__(actorArgs)
            self.__critic_building__(criticArgs)
            self.__expert_building__(expertArgs)
            self.__student_building__(studentArgs)

            if self.debug_mess:
                print(f"Actor MLP: {self.actor_NN}")
                print(f"Critic MLP: {self.critic_NN}")
                print(f"Expert MLP: {self.expert_NN}")
                print(f"Student MLP: {self.student_NN}")

            # Action noise
            self.std = nn.Parameter(actor_std_noise * torch.ones(actorArgs.outputs[0]))

        else:
            self.actor_NN = None
            self.critic_NN = None
            self.expert_NN = None
            self.student_NN = None

            if self.debug_mess:
                print(f"Actor critic and expert will be loaded.")

            self.std = None

        self.distribution = None

        # For optimizing the process
        Normal.set_default_validate_args = False

    def get_weights(self):
        return [self.actor_NN, self.critic_NN, self.expert_NN, self.student_NN, self.std]

    def load_weights(self, actor_critic):
        try:
            self.actor_NN, self.critic_NN, self.expert_NN, self.student_NN, self.std = actor_critic
        except Exception as e:
            self.actor_NN, self.critic_NN, self.expert_NN, self.std = actor_critic

        if self.debug_mess:
            print(f"Actor MLP: {self.actor_NN}")
            print(f"Critic MLP: {self.critic_NN}")
            print(f"Expert MLP: {self.expert_NN}")
            print(f"Student MLP: {self.student_NN}")

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

    def _scale_output(self, selected_action):
        return self.range_scaled * selected_action + self.min_range

    def update_distribution(self, observations, expert_observations):
        latent_space = self.expert_NN(expert_observations)
        selected_action = self.actor_NN(torch.cat((observations, latent_space), dim=-1))

        if self.scale_output:
            selected_action = self._scale_output(selected_action)

        self.distribution = Normal(selected_action, self.std)

    def act(self, observations, expert_observations):
        self.update_distribution(observations, expert_observations)
        return self.distribution.sample()

    def act_student(self, observations, history):
        latent_space = self.student_NN(history)
        selected_action = self.actor_NN(torch.cat((observations, latent_space), dim=-1))

        if self.scale_output:
            selected_action = self._scale_output(selected_action)

        return selected_action

    def act_test(self, observations, expert_observations):
        latent_space = self.expert_NN(expert_observations)
        selected_action = self.actor_NN(torch.cat((observations, latent_space), dim=-1))
        return selected_action

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, exp_obs):
        latent_space = self.expert_NN(exp_obs)
        return self.critic_NN(torch.cat((critic_observations, latent_space), dim=-1))

    def __default_values_kwargs__(self):
        self.debug_mess = False
        self.test = False
        self.range_scaled = 0.
        self.min_range = 0.

    def __prepare_kwargs__(self, kwargs):
        not_accepted, accepted = check_if_args_in_accepted(kwargs)

        if len(not_accepted):
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + not_accepted)

        if "debug_mess" in accepted:
            self.debug_mess = accepted["debug_mess"]

        if "test" in accepted:
            self.test = accepted["test"]

        if ("scale_max" in accepted and not "scale_min") or ("scale_min" in accepted and not "scale_max"):
            raise Exception("Scale without enough information. scale_min and scale_max are required")
        elif "scale_max" in accepted:
            self.scale_output = True
            self.range_scaled = accepted["scale_max"] - accepted["scale_min"]
            self.min_range = accepted["scale_min"]

    def __student_building__(self, studentArgs):
        if self.debug_mess:
            print("Starting to build the Student")

        layers = self.__generic_MLP_building__(studentArgs)

        if self.debug_mess:
            print("Creating the Student ...", end='  ')

        self.student_NN = nn.Sequential(*layers)

        if self.debug_mess:
            print("Done")

    def __expert_building__(self, expertArgs):
        if self.debug_mess:
            print("Starting to build the Expert")

        layers = self.__generic_MLP_building__(expertArgs)

        if self.debug_mess:
            print("Creating the Expert ...", end='  ')

        self.expert_NN = nn.Sequential(*layers)

        if self.debug_mess:
            print("Done")

    def __actor_building__(self, actorArgs):
        if self.debug_mess:
            print("Starting to build the Actor")

        layers = self.__generic_MLP_building__(actorArgs, scale_output=self.scale_output)

        if self.debug_mess:
            print(f"Creating the Actor{' with scaled output' if self.scale_output else ''} ...", end='  ')

        self.actor_NN = nn.Sequential(*layers)

        if self.debug_mess:
            print("Done")

    def __critic_building__(self, criticArgs):
        if self.debug_mess:
            print("Starting to build the Critic")

        layers = self.__generic_MLP_building__(criticArgs)

        if self.debug_mess:
            print("Creating the Critic ...", end='  ')

        self.critic_NN = nn.Sequential(*layers)

        if self.debug_mess:
            print("Done")

    @staticmethod
    def __generic_MLP_building__(args, scale_output=False):
        activation = get_activation(args.activation)

        layers = []
        size_input = args.inputs[0]

        for h in range(len(args.hidden_dim)):
            layers.append(nn.Linear(size_input, args.hidden_dim[h]))
            # nn.init.normal_(layers[-1].weight, mean=0.0, std=0.0)

            if not (activation is None):
                layers.append(activation)

            size_input = args.hidden_dim[h]

        layers.append(nn.Linear(size_input, args.outputs[0]))

        if scale_output:
            layers.append(get_activation("sigmoid"))
        # nn.init.normal_(layers[-1].weight, mean=0.0, std=0.0)

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
