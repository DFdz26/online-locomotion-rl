'''
Class: RBF
created by: arthicha srisuchinnawong
Modified by: Daniel Fernandez
e-mail: dafer21@student.sdu.dk
date last modification: 17-12-22

Radial Basis Function Network
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
from sklearn.cluster import KMeans
import torch as T
from torch.autograd import Variable
from torch.distributions import Normal, Categorical


# modular network
from modules.torchNet import torchNet

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------

# ------------------- class RBF ---------------------

class RBF(torchNet):	

	# -------------------- constructor -----------------------
	# (private)

	def __init__(self, cpg, n_state=1, sigma=0.01, tinit=100, device=None):

		# initialize network hyperparameter
		super().__init__(device)

		# device
		if device is not None:
			self.device = device
		elif torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.cpg = cpg
		self.__n_state = n_state
		self.__t_init = tinit
		self.__sigma = sigma

		# rbf
		self.__centers = self.__get_rbfcenter()

	def __get_rbfcenter(self):

		# step CPG
		unsorted_cpg = self.zeros(2,self.__t_init)
		for i in range(self.__t_init):
			cpg = self.cpg()
			for c in range(2):
				unsorted_cpg[c,i] = cpg[c]
		kdata = unsorted_cpg.cpu().numpy().transpose()

		# clustering the RBF center
		kmeans = KMeans(n_clusters=self.__n_state)
		kmeans.n_init = 10
		kmeans.fit(kdata)
		clus = kmeans.cluster_centers_.transpose()
		srtidx = np.argsort(clus[0,:])
		clus = clus[:,srtidx]
		return torch.FloatTensor(clus).to(self.device)

	
	# -------------------- handle functions -----------------------
	# (public)

	def forward(self,x):
		rbf = torch.unsqueeze(torch.exp(-(torch.pow(x[0] - self.__centers[0],2)+torch.pow(x[1] - self.__centers[1],2))/self.__sigma),1)
		return rbf.t()




	
		




