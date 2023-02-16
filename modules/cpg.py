'''
Class: CPG
created by: arthicha srisuchinnawong
e-mail: arsri21@student.sdu.dk
date: 1 August 2022

Central Pattern Generator
'''


# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import colorama 
colorama.init(autoreset=True)
from colorama import Fore

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import torch as T
from torch.autograd import Variable
from torch.distributions import Normal, Categorical

# modular network
from modules.torchNet import torchNet

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------

# ------------------- class CPG ---------------------

class CPG(torchNet):	

	# -------------------- constructor -----------------------
	# (private)

	def __init__(self,cpg_gamma=1.01,cpg_phi=0.5,tinit=100, device=None):

		# initialize network hyperparameter
		super().__init__(device)

		self.__t_init = tinit

		# CPG
		self.__cpg = self.torch(np.array([[0.0],[0.2]]))
		cpgweight = [[np.cos(cpg_phi),np.sin(cpg_phi)],[-np.sin(cpg_phi),np.cos(cpg_phi)]]
		self.__cpgweights = self.torch(cpg_gamma*np.array(cpgweight))

		self.reset()
	
	# -------------------- handle functions -----------------------
	# (public)

	def reset(self):
		self.__cpg = self.torch(np.array([[0.0],[0.2]]))
		for i in range(self.__t_init):
			self.forward()


	def forward(self):
		self.__cpg = torch.tanh(torch.matmul(self.__cpgweights,self.__cpg))
		return self.__cpg




	
		




