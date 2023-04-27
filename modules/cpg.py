"""
Class: CPG
created by: arthicha srisuchinnawong
Modified by: Daniel Fernandez
e-mail: dafer21@student.sdu.dk
date last modification: 5-01-23

Central Pattern Generator
"""

# ------------------- import modules ---------------------

import numpy as np  # cpu array
import torch  # cpu & gpu array

# modular network
from modules.torchNet import torchNet


# ------------------- configuration variables ---------------------

# ------------------- class CPG ---------------------

class CPG(torchNet):

    # -------------------- constructor -----------------------
    # (private)

    def __init__(self, cpg_gamma=1.01, cpg_phi=0.5, tinit=100, device=None, internal_amplitude=0.2):
        # initialize network hyperparameter
        super().__init__(device)
        self.__cpgweights = None
        self.gamma = cpg_gamma
        self.phi = cpg_phi

        self.__t_init = tinit
        self.internal_amplitude = internal_amplitude

        # CPG
        self.__cpg = self.torch(np.array([[0.0], [self.internal_amplitude]]))
        self.prev_out = self.__cpg.detach().clone()
        self.set_cpg_weight(self.phi, self.gamma)

    # -------------------- handle functions -----------------------
    # (public)

    def set_cpg_weight(self, cpg_phi, cpg_gamma, reset=True, t_time=None):
        cpgweight = [[np.cos(cpg_phi), np.sin(cpg_phi)], [-np.sin(cpg_phi), np.cos(cpg_phi)]]
        self.__cpgweights = self.torch(cpg_gamma * np.array(cpgweight))

        if reset:
            self.reset(t_time=t_time)

    def reset(self, t_time=None):
        if t_time is None:
            t_time = self.__t_init

        self.__cpg = self.torch(np.array([[0.0], [self.internal_amplitude]]))
        # self.prev_out = self.__cpg.detach().clone()

        for i in range(t_time):
            self.forward()

    def get_past_output(self):
        return self.prev_out

    def forward(self):
        self.prev_out = self.__cpg.detach().clone()
        self.__cpg = torch.tanh(torch.matmul(self.__cpgweights, self.__cpg))
        return self.__cpg
