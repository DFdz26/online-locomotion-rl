"""
Class: MN
created by: arthicha srisuchinnawong
Modified by: Daniel Fernandez
e-mail: dafer21@student.sdu.dk
date last modification: 5-01-23

Motor Neuron
"""

# ------------------- import modules ---------------------

# standard modules
import os
import os.path
import pickle

# math-related modules
import torch  # cpu & gpu array

# modular network
from modules.torchNet import torchNet

# plot

# ------------------- configuration variables ---------------------

# ------------------- class MN ---------------------
'''
motor neurons: M[t] = 2.0*tanh(k W B[t])
'''

cacheFileName = "cache"
caccheExtension = ".pickle"


class MN(torchNet):

    # -------------------- constructor -----------------------
    # (private)

    def __init__(self, hyperparams, outputgain=None, bias=False, device=None, dimensions=1, load_cache=True,
                 abs_weights=False, noise_to_zero=False):

        # initialize network hyperparameter
        super().__init__(device)
        self.folder_cache = "cache_init_values"

        # device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = torch.device('cpu')

        self.__n_state = hyperparams["n_state"]
        self.__n_out = hyperparams["n_out"]
        self.encoding = hyperparams["encoding"]
        self.__n_bias = 1 if bias else 0
        self.dimensions = dimensions

        if not os.path.exists(self.folder_cache):
            self.create_folder()

        self.full_cacheFilename = os.path.join(self.folder_cache, cacheFileName + "_" + self.encoding + caccheExtension)

        # initialize connection weight
        # self.W = self.zeros(self.__n_state+self.__n_bias,self.__n_out,grad=True)
        if os.path.exists(self.full_cacheFilename) and load_cache:
            self._load_cache_()
        else:
            self.W = 2 * torch.rand(self.__n_state + self.__n_bias, self.__n_out, requires_grad=False,
                                    device=self.device) - 1

            if load_cache:
                self._create_cache_()

        if abs_weights:
            self.W = torch.abs(self.W)

        self.Wn = torch.zeros((dimensions, self.__n_state + self.__n_bias, self.__n_out), device=self.device)
        self.Wn = torch.add(self.W, self.Wn)

        self.reset()

        # normalize all joints -> output gain

        self.__output_gain = self.zeros(1, self.__n_out * dimensions) + 1.0 if outputgain is None else self.torch(
            outputgain)
        self.__output_gain = torch.reshape(self.__output_gain, (dimensions, self.__n_out))

    def create_folder(self):

        os.mkdir(self.folder_cache)

    def _load_cache_(self):
        print("loaded cache")
        with open(self.full_cacheFilename, 'rb') as f:
            self.W = pickle.load(f)

    def _create_cache_(self):

        print("stored cache")
        with open(self.full_cacheFilename, 'wb') as f:
            pickle.dump(self.W, f)

    def load_weights(self, nw, update_Wn=True):
        print("aaaaaaaaaaaaa")
        self.device = 'cuda:0'
        print(nw)
        print(self.device)
        W_init = torch.FloatTensor(nw.to('cpu')).to(self.device)
        print(W_init)
        print(self.W.shape)
        self.W = torch.reshape(W_init, self.W.shape)

        if update_Wn:
            self.Wn = torch.zeros((self.dimensions, self.__n_state + self.__n_bias, self.__n_out), device=self.device)
            self.Wn = torch.add(self.W, self.Wn)

    def get_w_arr_(self):
        return self.W.squeeze()

    # -------------------- set values -----------------------
    # (public)

    def apply_noise_tensor(self, noise):
        # print(f"noise {noise}")
        noise_a = torch.reshape(noise, self.Wn.shape)
        self.Wn = self.W + noise_a

    def apply_noise_np(self, noise):
        print(f"noise {noise}")
        noise_a = torch.FloatTensor(noise).to(self.device)
        noise_a = torch.reshape(noise_a, self.Wn.shape)
        self.Wn = self.W + noise_a

    def apply_noise(self, noise):
        self.Wn = self.W + noise

    # -------------------- handle functions -----------------------
    # (public)

    def reset(self):
        # reset connection
        pass

    def forward(self, x):
        if self.__n_bias > 0:
            shape = list(x.shape)
            shape[-1] = 1
            x1 = torch.cat([x, torch.ones(shape).to(self.device)], dim=-1)
        else:
            x1 = x
        outputs = 1 * torch.tanh((x1 @ self.Wn)).reshape(self.dimensions, self.__n_out)
        return outputs
