"""
Class: CPGRBFN
created by: arthicha srisuchinnawong
Modified by: Daniel Fernandez
e-mail: dafer21@student.sdu.dk
date last modification: 5-01-23

CPG-RBFN class for locomotion learning
"""

# ------------------- import modules ---------------------

# standard modules
import time, sys, os

# math-related modules
import numpy as np  # cpu array
import torch  # cpu & gpu array
from torch.autograd import Variable
from torch.distributions import Normal

# network
# from modules.utils import HyperParams # hyperparameter structure
from modules.torchNet import torchNet  # torch-based network template
from modules.cpg import CPG  # Central Pattern Generator
from modules.rbf import RBF  # Central Pattern Generator
from modules.mn import MN  # Motor Neurons
from modules.utils import CPGUtils  # Motor Neurons

key_direct = "direct"
key_indirect = "indirect"
key_semi_indirect = "semi_indirect"


# ------------------- class CPGRBFN ---------------------

class CPGRBFN(torchNet):

    # ---------------------- constructor ------------------------
    def __init__(self, config, dimensions=1, load_cache=True):

        self.device = config["device"]
        self.reversed_cpg = False
        self.__n_out = 0
        self.dimensions = dimensions

        super().__init__(config["device"])
        self.index_aux = torch.arange(self.dimensions, device=self.device, requires_grad=False)

        # ---------------------- initialize modular neural network ------------------------
        # (update in this order)

        self._set_up_hyperparameters(config['HYPERPARAM'])

        # motor gain
        if "MN" in config:
            mn_gain = list((config['MN']['GAIN']).split(","))
            mn_gain = [float(gain) for gain in mn_gain]
        else:
            mn_gain = None

        abs_weights = config['abs_weights'] if 'abs_weights' in config else False

        # ---------------------- initialize modular neural network ------------------------
        # (update in this order)

        # CPG
        self.cpg = CPG(cpg_gamma=float(config['CPG']['GAMMA']), cpg_phi=float(config['CPG']['PHI']),
                       tinit=self.__t_init, device=self.device)

        # RBF
        self.rbf = RBF(self.cpg, self.__n_state, sigma=float(config['RBF']['SIGMA']), tinit=self.__t_init,
                       device=self.device)

        # Motor Network
        self.mn = MN(self.__hyperparams, outputgain=mn_gain, device=self.device, dimensions=dimensions,
                     load_cache=load_cache, abs_weights=abs_weights)

        self.cpg_history = CPGUtils(config, self.cpg, verbose=True)

        # ---------------------- initialize neuron activity ------------------------
        self.inputs = self.zeros(1, self.__n_in)
        self.outputs = self.zeros(dimensions, self.__n_motors)

        # initialize cpg activity
        self.reset()
        self.cpg_history.init_buffers()

        self.CPG_period = self.cpg_history.period

    def get_weights(self):
        return self.mn.W

    def load_weights(self, weights):
        self.mn.load_weights(weights)

    def pretraing_process(self, pibb):
        noise = pibb.get_noise()
        self.mn.apply_noise_tensor(noise)

    def modify_weights(self, weights):
        self.mn.W = weights

    def train_modify_weights(self, pibb, rewards):
        self.modify_weights(pibb.step(rewards, self.get_weights()))

    def get_n_outputs(self):
        return self.__n_out

    def _set_up_hyperparameters(self, in_dic):
        self.__n_in = int(in_dic['NIN'])
        self.__n_state = int(in_dic['NSTATE'])
        self.__t_init = int(in_dic['TINIT'])
        self.__motors_per_leg = int(in_dic['MOTORS_LEG'])
        self.__n_leg = int(in_dic['NLEG'])
        self.__n_motors = self.__motors_per_leg * self.__n_leg
        self.__encoding = in_dic["ENCODING"]

        if 0 != self.__n_leg % 2:
            raise Exception("Number of legs needs to be pair")

        self.__hyperparams = {
            "n_out": 0,
            "n_in": self.__n_in,
            "n_state": self.__n_state,
            "encoding": self.__encoding,
        }

        dic_converter = {
            key_direct: self.__output_from_direct_encoding,
            key_indirect: self.__output_from_indirect_encoding,
            key_semi_indirect: self.__output_from_semi_indirect_encoding,
        }

        if self.__encoding in dic_converter:
            dic_converter[self.__encoding]()
        else:
            aux_helper = ""
            for key in dic_converter:
                aux_helper += f"- {key}\n"

            raise Exception(f"{self.__encoding} is not a valid encoding. Implemented encodings: \n{aux_helper}.")

        self.__hyperparams["n_out"] = self.__n_out

        print(f"self.__n_out: {self.__n_out}")

    # ---------------------- debugging   ------------------------
    def get_state(self):
        return self.bf.detach().cpu().numpy()[0]

    def get_output(self):
        return self.outputs.detach().cpu().numpy()[0, :3]

    def get_cpg_delayed(self, steps_delayed):
        return self.cpg_history.read_stored(steps_delayed)

    def __output_from_direct_encoding(self):
        self.__n_out = self.__n_motors

    def __output_from_indirect_encoding(self):
        self.__n_out = self.__motors_per_leg
        self.aux_m = [0, 4, 2]
        self.aux_m_d = [3, 1, 5]

    def __output_from_semi_indirect_encoding(self):
        self.__n_out = self.__n_motors / 2

    def __resize_rbfn_to_n_motor(self, rbfn_out, rbfn_delayed_out):
        dic_converter = {
            key_direct: self.__rbfn_to_direct_encoding,
            key_indirect: self.__rbfn_to_indirect_encoding,
            key_semi_indirect: self.__rbfn_to_semi_indirect_encoding,
        }

        dic_converter[self.__encoding](rbfn_out, rbfn_delayed_out, self.reversed_cpg)

    def __rbfn_to_direct_encoding(self, rbfn_out, rbfn_delayed_out, reversed):
        normal = rbfn_out
        delayed = rbfn_delayed_out

        started_motor = 0

        for _ in range(int(self.__n_leg / 2)):
            self.outputs[self.index_aux, started_motor] = normal[self.index_aux, started_motor]
            self.outputs[self.index_aux, started_motor + 1] = delayed[self.index_aux, started_motor + 1]
            self.outputs[self.index_aux, started_motor + 2] = normal[self.index_aux, started_motor + 2]

            self.outputs[self.index_aux, started_motor + 3] = delayed[self.index_aux, started_motor + 3]
            self.outputs[self.index_aux, started_motor + 4] = normal[self.index_aux, started_motor + 4]
            self.outputs[self.index_aux, started_motor + 5] = delayed[self.index_aux, started_motor + 5]

            if reversed:
                aux_change = normal
                normal = delayed
                delayed = aux_change

            started_motor += 6

    def __rbfn_to_indirect_encoding(self, rbfn_out, rbfn_delayed_out, reversed):
        normal = rbfn_out
        delayed = rbfn_delayed_out

        started_motor = 0

        # for i in range(self.__n_out):
        # 	self.outputs[self.index_aux, [self.aux_m[i], self.aux_m[i] + 6]] = normal[self.index_aux, i]
        # 	self.outputs[self.index_aux, [self.aux_m_d[i], self.aux_m_d[i]]] = delayed[self.index_aux, i]

        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]
        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]
        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]

        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]
        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]
        # self.outputs[self.index_aux, [0, 6]] = normal[self.index_aux, 0]

        for _ in range(int(self.__n_leg / 2)):
            self.outputs[self.index_aux, started_motor] = normal[self.index_aux, 0]
            self.outputs[self.index_aux, started_motor + 1] = delayed[self.index_aux, 1]
            self.outputs[self.index_aux, started_motor + 2] = normal[self.index_aux, 2]

            self.outputs[self.index_aux, started_motor + 3] = delayed[self.index_aux, 0]
            self.outputs[self.index_aux, started_motor + 4] = normal[self.index_aux, 1]
            self.outputs[self.index_aux, started_motor + 5] = delayed[self.index_aux, 2]

            if reversed:
                aux_change = normal
                normal = delayed
                delayed = aux_change

            started_motor += 6

    def __rbfn_to_semi_indirect_encoding(self, rbfn_out, rbfn_delayed_out, reversed):
        normal = rbfn_out[0]
        delayed = rbfn_delayed_out[0]

        started_motor = 0
        started_rbfn = 0

        for _ in range(int(self.__n_leg / 2)):
            self.outputs[0][started_motor] = normal[started_rbfn]
            self.outputs[0][started_motor + 1] = delayed[started_rbfn + 1]
            self.outputs[0][started_motor + 2] = normal[started_rbfn + 2]

            self.outputs[0][started_motor + 3] = delayed[started_rbfn]
            self.outputs[0][started_motor + 4] = normal[started_rbfn + 1]
            self.outputs[0][started_motor + 5] = delayed[started_rbfn + 2]

            if reversed:
                aux_change = normal
                normal = delayed
                delayed = aux_change

            started_motor += 6
            started_rbfn += 3

    def get_cpg_output(self):
        return self.cpg_o

    # ---------------------- update   ------------------------
    def reset(self):
        # reset neural modules
        self.cpg.reset()
        self.mn.reset()

    def forward(self, x):

        # update cpg-rbf
        self.cpg_o = self.cpg()
        self.cpg_history.store_cpg_steps(self.cpg_o)

        self.bf = self.rbf(self.cpg_o)
        self.bf_delayed = self.rbf(self.get_cpg_delayed(int(0.5 * self.CPG_period)))

        # update motor neurons
        motor1 = self.mn(self.bf)
        motor2 = self.mn(self.bf_delayed)

        self.__resize_rbfn_to_n_motor(motor1, motor2)

        # self.outputs = motor1

        # for i in range(int(self.__n_out/6)):
        # 	# normal = motor1 if i % 2 == 0 else motor2
        # 	# delayed = motor2 if i % 2 == 0 else motor1
        # 	normal = motor1
        # 	delayed = motor2

        # 	self.outputs[0][0 + 6*i] = normal[0][0]
        # 	self.outputs[0][1 + 6*i] = delayed[0][1]
        # 	self.outputs[0][2 + 6*i] = normal[0][2]

        # 	self.outputs[0][3 + 6*i] = delayed[0][0]
        # 	self.outputs[0][3 + 6*i] = normal[0][1]
        # 	self.outputs[0][5 + 6*i] = delayed[0][2]

        # 	# self.outputs[0][0 + 6*i] = motor1[0][6*i]
        # 	# self.outputs[0][1 + 6*i] = motor2[0][1 + 6*i]
        # 	# self.outputs[0][2 + 6*i] = motor1[0][2 + 6*i]

        # 	# self.outputs[0][3 + 6*i] = motor2[0][6*i]
        # 	# self.outputs[0][3 + 6*i] = motor1[0][1 + 6*i]
        # 	# self.outputs[0][5 + 6*i] = motor2[0][2 + 6*i]

        return self.outputs

    @staticmethod
    def HyperParams(n_states, n_in, n_out):
        class _hyp:

            def __init__(self) -> None:
                self.n_state = None
                self.n_in = None
                self.n_out = None

        in_hyp = _hyp()
        in_hyp.n_state = n_states
        in_hyp.n_in = n_in
        in_hyp.n_out = n_out

        return in_hyp
