from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import math

from learningAlgorithm.PIBB.PIBB import PIBB
from isaacGymConfig.envConfig import EnvConfig
# from isaacGymConfig.RobotConfig import RobotConfig
from Runner import Runner

from modules.cpgrbfn2 import CPGRBFN
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards
from learningAlgorithm.PPO.ActorCritic import ActorCritic
from learningAlgorithm.PPO.ActorCritic import NNCreatorArgs
from learningAlgorithm.PPO.PPO import PPO

from learningAlgorithm.PIBB.PIBB import PIBB

from learningAlgorithm.CPG_MLP import MLP_CPG
from learningAlgorithm.CPG_MLP import PPO_PIBB
from isaacGymConfig.TerrainConfig import Terrain, TerrainComCfg
from isaacGymConfig.Curriculum import Curriculum

config_file = "models/configs/config_minicheeta.json"
graph_name = "graph_minicheeta_testing"

# checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/06_04_2023__07_25_57/2400.pickle"
# checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/06_04_2023__16_25_16/1200.pickle"
# checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/06_04_2023__22_07_23/300.pickle"
# checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/2023_04_11.07_15_29/600.pickle"
checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/2023_04_18.23_46_08/600.pickle"
# checkpoint_file = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/07_04_2023__00_48_51/400.pickle"

# config_file = "models/configs/config_b1.json"
# graph_name = "graph_b1_learning"

iterations_without_control = 1
SAVE_DATA = True
LOAD_CACHE = True
TERRAIN_CURRICULUM = True
rollouts = 10
num_env_colums = 10
learning_rate_PPO = 0.0000003  # 0.0000003
start_PPO_acting_iteration = 350
device = "cuda:0"

logger = Logger(test_value=True, size_figure=1)


def config_terrain(env_config):
    list_terrains = [
        # {
        #     "terrain": "flat_terrain",
        # },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.010,
            "max_height": 0.010,
            "step": 0.010,
            "downsampled_scale": 0.5
        },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.035,
            "max_height": 0.035,
            "step": 0.035,
            "downsampled_scale": 0.5
        },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.025,
            "downsampled_scale": 0.5
        },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.06,
            "max_height": 0.06,
            "step": 0.03,
            "downsampled_scale": 0.5
        },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.075,
            "max_height": 0.075,
            "step": 0.025,
            "downsampled_scale": 0.5
        }
    ]

    terrain_com_conf = TerrainComCfg()

    # Compute the number of needed columns
    aux = num_env_colums if rollouts > num_env_colums else rollouts
    x = aux * -env_config.spacing_env - terrain_com_conf.border_x * 2
    y = terrain_com_conf.border_y * 2 + env_config.spacing_env_x * math.ceil(rollouts / num_env_colums) + 1
    terrain_com_conf.columns = math.ceil(x / terrain_com_conf.terrain_length)
    terrain_com_conf.terrain_width = y/2
    terrain_obj = Terrain(device, rollouts, list_terrains, terrain_com_conf)

    
    return terrain_obj, None


def config_env():
    env_config.num_env = rollouts
    env_config.actions_scale = actions_scale
    env_config.hip_scale = hip_scale
    env_config.dt = dt
    env_config.num_env_colums = num_env_colums
    env_config.iterations_without_control = iterations_without_control


    env_config.cfg_observations.enable_observe_friction = False
    env_config.cfg_observations.enable_observe_restitution = False
    env_config.cfg_observations.enable_observe_motor_strength = True
    env_config.cfg_observations.enable_observe_payload = True

noise_boost = 1.75
dt = 0.005
seconds_iteration = 15
max_iterations = 5
step_env = int(seconds_iteration / dt)
step_env = int(seconds_iteration / 0.01)

show_final_graph = False

# For tracking the distance
reward_list = {
    "x_distance": {
        "weight": 0.,
        "reward_data" : {
            "absolute_distance": False
        }
    }
}


actions_scale = 0.2
hip_scale = 0.2

cpg_rbfn_information = logger.recover_nn_information(filename=checkpoint_file)
cpg_rbfn_information["HYPERPARAM"]['ENCODING'] = 'direct'
algorithm_param= logger.recover_algorithm_parameters(filename=checkpoint_file)
cpg_rbf_nn = CPGRBFN(cpg_rbfn_information, dimensions=rollouts, noise_to_zero=True)

n_out = cpg_rbf_nn.get_n_outputs()
print(f"n_out: {n_out}")

actorCritic = ActorCritic(None, None, 0., None, debug_mess=True, test=True)
policy = MLP_CPG(actorCritic, cpg_rbf_nn)

ppo = PPO(actorCritic, device=device, verbose=True, loading=True)
ppo.cfg = algorithm_param[0]

reward_obj = Rewards(rollouts, device, reward_list, 0.999999, step_env, discrete_rewards=True)
pibb = PIBB(rollouts, 0, 1, 20 * n_out, 0, 0, device="cuda:0", boost_noise=noise_boost)


env_config = EnvConfig()

config_env()
terrain_obj, terrain_curr = config_terrain(env_config)
logged = False

learning_algorithm = PPO_PIBB(ppo, pibb, None)
learning_algorithm.read_data_point(checkpoint_file, logger, policy, test=True)
pibb.noise_arr.fill_(0)
pibb.policy.mn.apply_noise_tensor(pibb.noise_arr)
print("=======")
print(pibb.policy.mn.Wn[0])


robot = Runner(policy, learning_algorithm, logger, config_file, env_config, reward_obj, n_out,
               terrain_obj, curricula=None, verbose=True, store_observations=True)

try:
    robot.test_agent(max_iterations, step_env)
    logged = True
    logger.log(SAVE_DATA, show_final_graph, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
except KeyboardInterrupt:
    if not logged:
        logger.log(SAVE_DATA, True, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
