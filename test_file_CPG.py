from modules.cpg import CPG
from modules.hopf_oscillators import HopfOscillators
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from modules.cpgrbfn2 import CPGRBFN

def build_cpg_signal2(cpg_o, previous_cpg, natural_amplitude):

    return natural_amplitude * math.cos(math.pi/2 * cpg_o/natural_amplitude)
    going_up = cpg_o > previous_cpg
    nat = -natural_amplitude * math.cosh(cpg_o/natural_amplitude)

    if going_up:
        if cpg_o > 0:
            out = -nat + cpg_o
        else:
            out = -cpg_o - nat

        return out

    if cpg_o > 0:
        out = nat - cpg_o
    else:
        out = nat + cpg_o

    return out

if __name__ == "__main__":
    # init_amplitude = 0.2
    # init_phase = 0.2
    # intrinsic_frequency = 0.01
    # intrinsic_amplitude = 0.2
    # command_signal_a = 1.0
    # command_signal_d = 1
    #
    # cpg = CPG(1.01, 0.06, 500)
    # cpg_hopf = HopfOscillators(init_amplitude, init_phase, intrinsic_frequency, intrinsic_amplitude, command_signal_a,
    #                       command_signal_d)
    #
    # xpoints = []
    # cpg_out = []
    # cpg_out_2 = []
    # cpg_out_2_computed = []
    # cpg_inv = []
    # multipl = 1.
    #
    # for i in range(500):
    #     out_cpgs = cpg.forward()
    #     prev_out_cpgs = cpg.get_past_output()
    #
    #     computed = build_cpg_signal2(float(out_cpgs[0, 0]), float(prev_out_cpgs[0, 0]), cpg.internal_amplitude)
    #
    #     cpg_out_2_computed.append(-computed)
    #     cpg_out.append(float(out_cpgs[0, 0]))
    #     cpg_inv.append(-float(out_cpgs[0, 0]))
    #     cpg_out_2.append(float(out_cpgs[1, 0]))
    #     xpoints.append(i)
    #
    #     if i == 500:
    #         cpg_phi = 0.1
    #         cpg_gamma = 1.01
    #         cpg.set_cpg_weight(cpg_phi, cpg_gamma, reset=False)
    #
    # xpoints = np.array(xpoints)
    # cpg_out = np.array(cpg_out)
    #
    # plt.plot(cpg_out, 'g')
    # plt.plot(cpg_out_2_computed, 'b')
    # # print(cpg_out_2_computed)
    # plt.plot(cpg_out_2, 'r')
    # plt.show()

    hyperparam = {
        "NIN": 1,
        "NSTATE": 20,
        "MOTORS_LEG": 3,
        "NLEG": 4,
        "TINIT": 1000,
        "ENCODING": "direct"
    }

    # cpg_param = {
    #     "TYPE": "SO2",
    #     "PARAMETERS": {
    #         "GAMMA": 1.01,
    #         "PHI": 0.06
    #     },
    # }

    cpg_param = {
        "TYPE": "hopf",
        "PARAMETERS": {
            "INIT_AMPLITUDE": 0.2,
            "INIT_PHASE": 0.0,
            "INTRINSIC_FREQUENCY": 1.0,
            "INTRINSIC_AMPLITUDE": 0.2,
            "COMMAND_SIGNAL_A": 1.0,
            "COMMAND_SIGNAL_D": 1.0,
            "EXPECTED_DT": 0.005 * 4,
        },
    }

    cpg_utils = {
        "STEPS_INIT": 250,
        "SHOW_GRAPHIC": False,
    }

    rbf_param = {
        "SIGMA": 0.04,
    }

    config = {
        "device": "cpu",
        "HYPERPARAM": hyperparam,
        "RBF": rbf_param,
        "CPG": cpg_param,
        "UTILS": cpg_utils
    }

    cpg_rbf_nn = CPGRBFN(config, dimensions=1, load_cache=False)
    cpg_rbfn_output_1 = []
    cpg_rbfn_output_2 = []
    cpg_rbfn_output_3 = []
    frequency_change = 0.

    for i in range(100):

        if i == 50:
            frequency_change = 10.0
        elif i == 70:
            frequency_change = 0.0

        cpg_rbf_nn.forward(frequency_change=frequency_change)
        x = cpg_rbf_nn.get_output()
        cpg_rbfn_output_1.append(x[0])
        cpg_rbfn_output_2.append(x[1])
        cpg_rbfn_output_3.append(x[2])
        print(x)
    print(cpg_rbf_nn.CPG_period)

    plt.plot(cpg_rbfn_output_1, 'o')
    plt.plot(cpg_rbfn_output_2, 'x')
    # plt.plot(cpg_rbfn_output_3, 'g')
    plt.show()