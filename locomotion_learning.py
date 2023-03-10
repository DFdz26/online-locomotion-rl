from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch


from learningAlgorithm.PIBB import PIBB
from isaacGymConfig.envConfig import EnvConfig
from isaacGymConfig.RobotConfig import RobotConfig

from modules.cpgrbfn2 import CPGRBFN
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards

config_file = "configs/config_minicheeta.json"
graph_name = "graph_minicheeta_learning"

# config_file = "configs/config_b1.json"
# graph_name = "graph_b1_learning"

SAVE_DATA = False
LOAD_CACHE = False

reward_list = {
    "x_distance": {
        "weight": 1.5,
        "reward_data" : {
            "absolute_distance": False
        }
    },

    "y_distance": {
        "weight": -1.,
        "reward_data" : {
            "absolute_distance": True
        }
    },

    "stability": {
        "weight": -1.,
        "reward_data" : {
            "absolute_distance": False,
            "weights": {
                "std_height": 1.3,
                "mean_x_angle": 1.3,
                "mean_y_angle": 1.1,
                "mean_z_angle": 1.3,
                "distance": 0.5,
            }
        }
    },

    "high_penalization_contacts": {
        "weight": -1.,
        "reward_data" : {
            "absolute_distance": False,
            "max_clip": 2.5,
            "weights": {
                "correction_state": 0.02,
                "distance": 0.5,
            }
        }
    },
}

n_kernels = 20
variance = 0.036
decay = 0.992
h = 10
rollouts = 15
noise_boost = 1.5

show_final_graph = True

encoding = "indirect"
# encoding = "direct"

actions_scale = 0.2
hip_scale = 0.2

hyperparam = {
    "NIN": 1,
    "NSTATE": 20,
    "MOTORS_LEG": 3,
    "NLEG": 4,
    "TINIT": 1000,
    "ENCODING": encoding
}

cpg_param = {
    "GAMMA": 1.01,
    "PHI": 0.06
}

cpg_utils = {
    "STEPS_INIT": 250,
    "SHOW_GRAPHIC": False,
}


rbf_param = {
    "SIGMA": 0.04,
}

config = {
    "device": "cuda",
    "HYPERPARAM" : hyperparam,
    "RBF": rbf_param,
    "CPG": cpg_param,
    "UTILS": cpg_utils
}
cpg_rbf_nn = CPGRBFN(config, dimensions=rollouts, load_cache=LOAD_CACHE)

n_out = cpg_rbf_nn.get_n_outputs()
print(f"n_out: {n_out}")

reward_obj = Rewards(rollouts, "cuda:0", reward_list)
pibb = PIBB(rollouts, h, 1, n_kernels*n_out, decay, variance, device="cuda:0", boost_noise=noise_boost)
logger = Logger(save=SAVE_DATA, frequency=10, PIBB_param=pibb.get_hyper_parameters(), nn_config=config)
env_config = EnvConfig()

def config_env ():
    env_config.num_env = rollouts
    env_config.actions_scale = actions_scale
    env_config.hip_scale = hip_scale

config_env()
logged = False

robot = RobotConfig(config_file, env_config, cpg_rbf_nn, pibb, logger, reward_obj)

try:
    robot.run_con(True)
    logged = True
    logger.log(SAVE_DATA, show_final_graph, plot_file_name=graph_name)
except KeyboardInterrupt:
    if not logged:
        logger.log(SAVE_DATA, True, plot_file_name=graph_name)

