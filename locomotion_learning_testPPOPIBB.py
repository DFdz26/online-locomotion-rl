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
from isaacGymConfig.Curriculum import Curriculum, TerrainCurrCfg, AlgorithmCurrCfg, RandomizationCurrCfg

config_file = "models/configs/config_minicheeta.json"
graph_name = "graph_minicheeta_learning"
cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/06_04_2023__22_07_23/300.pickle"

# config_file = "models/configs/config_b1.json"
# graph_name = "graph_b1_learning"

ACTIVE_RECORDING_CAMERA = True
frequency_recording = 100

CURRICULUM_CPG_RBFN = True
RENDER_GUI = False
SAVE_DATA = True
RECOVER_CPG = False
LOAD_CACHE = True
TERRAIN_CURRICULUM = True
# rollouts = 1500
rollouts = 1500
iterations_without_control = 1
num_env_colums = 100
# learning_rate_PPO = 0.0000003  # 0.0000003
start_PPO_acting_iteration = 350
device = "cuda:0"

if RECOVER_CPG:
    start_PPO_acting_iteration = 1


def config_camera(activate, _env_config: EnvConfig, _logger: Logger, frames, fps=30):
    if activate:
        from modules.logger import VideoSettings

        n_camera = 3
        height_camera = 480
        width_camera = 640

        _env_config.sensors.Activations.camera_activated = True
        _env_config.sensors.Camera.height = height_camera
        _env_config.sensors.Camera.width = width_camera
        _env_config.sensors.Camera.n_camera = n_camera

        settings = []
        for _ in range(n_camera):
            vSett = VideoSettings()
            vSett.width = width_camera
            vSett.height = height_camera
            vSett.n_frames = frames
            vSett.fps = fps

            settings.append(vSett)
        
        settings[0].filename = 'easy'
        settings[1].filename = 'medium'
        settings[2].filename = 'hard'

        _logger.load_multiple_video_recoder(settings, frequency_recording)


def config_learning_curriculum():
    algCfg = AlgorithmCurrCfg()
    algCfg.PIBBCfg.threshold = start_PPO_acting_iteration
    algCfg.PPOCfg.gamma = 0.5
    algCfg.PPOCfg.change_RW_scales = True

    if CURRICULUM_CPG_RBFN and not RECOVER_CPG:
        algCfg.PIBBCfg.switching_indirect_to_direct = True
        algCfg.PIBBCfg.threshold_switching = 100
        algCfg.PIBBCfg.decay_at_switching = 0.992
        algCfg.PIBBCfg.variance_at_switching = 0.009
        algCfg.PIBBCfg.boost_first_switching_noise = 1.
        algCfg.PIBBCfg.change_RW_scales_when_switching = True

    return algCfg


def config_randomization_curriculum():
    randCurrCfg = RandomizationCurrCfg()

    randCurrCfg.MotorParameters.randomize_motor_strength = True
    randCurrCfg.MotorParameters.step_randomization_motor = 0.

    randCurrCfg.ModelParameters.randomize_payload = True
    randCurrCfg.ModelParameters.step_randomization_payload = 0.

    randCurrCfg.ModelParameters.randomize_friction = True
    randCurrCfg.ModelParameters.step_randomization_friction = 0.

    randCurrCfg.ModelParameters.randomize_restitution = True
    randCurrCfg.ModelParameters.step_randomization_restitution = 0.

    randCurrCfg.Control.randomization_activated = True
    randCurrCfg.Control.generate_first_randomization = True
    randCurrCfg.Control.start_randomization_iteration = start_PPO_acting_iteration
    randCurrCfg.Control.randomization_interval_iterations = 3

    return randCurrCfg


def config_terrain(env_config):
    list_terrains = [
        {
            "terrain": "flat_terrain",
        },
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
        # {
        #     "terrain": "random_uniform_terrain",
        #     "min_height": -0.075,
        #     "max_height": 0.075,
        #     "step": 0.025,
        #     "downsampled_scale": 0.5
        # }
    ]

    terrain_com_conf = TerrainComCfg()

    # Compute the number of needed columns
    aux = num_env_colums if rollouts > num_env_colums else rollouts
    x = aux * -env_config.spacing_env - terrain_com_conf.border_x * 2
    y = terrain_com_conf.border_y * 2 + env_config.spacing_env_x * math.ceil(rollouts / num_env_colums) + 1
    terrain_com_conf.columns = math.ceil(x / terrain_com_conf.terrain_length)
    terrain_com_conf.terrain_width = y
    terrain_obj = Terrain(device, rollouts, list_terrains, terrain_com_conf)

    if TERRAIN_CURRICULUM:
        curriculum_terr = TerrainCurrCfg()

        curriculum_terr.object = terrain_obj
        first_curr = start_PPO_acting_iteration + 70
        second_curr = start_PPO_acting_iteration + 150
        third_curr = start_PPO_acting_iteration + 151
        fourth_curr = start_PPO_acting_iteration + 152  # 700
        fifth_curr = start_PPO_acting_iteration + 500  # 1750
        curriculum_terr.Control.threshold = [first_curr, second_curr, third_curr, fourth_curr, fifth_curr]
        curriculum_terr.percentage_step = 0.32
    else:
        curriculum_terr = None

    return terrain_obj, curriculum_terr


def config_env():
    env_config.num_env = rollouts
    env_config.actions_scale = actions_scale
    env_config.hip_scale = hip_scale
    env_config.dt = dt
    env_config.num_env_colums = num_env_colums
    env_config.render_GUI = RENDER_GUI
    env_config.iterations_without_control = iterations_without_control

    env_config.cfg_observations.enable_observe_friction = False
    env_config.cfg_observations.enable_observe_restitution = False
    env_config.cfg_observations.enable_observe_motor_strength = True
    env_config.cfg_observations.enable_observe_payload = True


reward_list = {
    "x_distance": {
        "weight": 2.2 * 1,
        "reward_data": {
            "absolute_distance": False
        }
    },

    # "y_distance": {
    #     "weight": -0.06,
    #     "reward_data": {
    #         "absolute_distance": True
    #     }
    # },

    # "stability": {
    #     "weight": -1.1,
    #     "reward_data": {
    #         "absolute_distance": True,
    #         "weights": {
    #             "std_height": 1.3,
    #             "mean_x_angle": 1.3,
    #             "mean_y_angle": 1.1,
    #             "mean_z_angle": 1.3,
    #             "distance": 0.5,
    #         }
    #     }
    # },

    "high_penalization_contacts": {
        "weight": -0.25,
        "reward_data": {
            "absolute_distance": True,
            "max_clip": 2.5,
            "weights": {
                "correction_state": 0.02,
                "distance": 0.5,
            }
        }
    },

    "height_error": {
        "weight": -2.9,
        "reward_data": {
            "max_clip": 2.5,
        }
    },

    "slippery": {
        "weight": 1.,
        "reward_data": {
            "slippery_coef": -0.0088,
        }
    },

    "smoothness": {
        "weight": 1.,
        "reward_data": {
            "jerk_coef": -0.00002,
        }
    },

    "z_vel": {
        "weight": 0.1,
        "reward_data": {
            "exponential": False,
            "weight": -0.24
        }
    },

    "roll_pitch": {
        "weight": 0.077,
        "reward_data": {
            "exponential": False,
            "weight": -0.15
        }
    },

    "yaw_vel": {
        "weight": 0.028,
        "reward_data": {
            "exponential": False,
            "weight": -0.1,
            "command": 0.,
        }
    },

    "y_velocity": {
        "weight": 0.1,
        "reward_data": {
            "exponential": False,
            "weight": -0.075  # 0.05
        }
    },

    "x_velocity": {
        "weight": 1. * 2 * 1,
        "reward_data": {
            "exponential": False,
            "weight": 0.178  # 0.177
        }
    },

    "vel_cont": {
        "weight": -0.2,  # 0.25
    },

    # "orthogonal_angle_error": {
    #     "weight": 0.1,
    #     "reward_data": {
    #         "weight": -0.02
    #     }
    # },

}

n_kernels = 20
h = 10

if RECOVER_CPG:
    decay = 0.8
    variance = 0.003
    noise_boost = 0.9
elif CURRICULUM_CPG_RBFN:
    decay = 0.992
    variance = 0.022
    noise_boost = 1.75
else:
    variance = 0.027
    decay = 0.9965
    noise_boost = 1.75

dt = 0.005
seconds_iteration = 5 / 2
max_iterations = 99001
step_env = int(seconds_iteration / dt)
step_env = int(seconds_iteration / 0.01)

show_final_graph = True

if CURRICULUM_CPG_RBFN:
    encoding = "indirect"
else:
    # encoding = "indirect"
    encoding = "direct"

actions_scale = 0.2
hip_scale = 0.2

hyperparam = {
    "NIN": 1,
    "NSTATE": n_kernels,
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
    "HYPERPARAM": hyperparam,
    "RBF": rbf_param,
    "CPG": cpg_param,
    "UTILS": cpg_utils
}

cpg_rbf_nn = CPGRBFN(config, dimensions=rollouts, load_cache=LOAD_CACHE)

n_out = cpg_rbf_nn.get_n_outputs()
print(f"n_out: {n_out}")

latent_space_size = 12
priv_obs = 21

actorArgs = NNCreatorArgs()
# actorArgs.inputs = [39]
actorArgs.inputs = [45 + latent_space_size]
# actorArgs.hidden_dim = [128, 64]
# actorArgs.hidden_dim = [256, 128]
actorArgs.hidden_dim = [512, 256]
actorArgs.outputs = [n_out if not CURRICULUM_CPG_RBFN else 12]

criticArgs = NNCreatorArgs()
criticArgs.inputs = [45 + latent_space_size]
# criticArgs.hidden_dim = [128, 64]
criticArgs.hidden_dim = [512, 256]
criticArgs.outputs = [1]

expertArgs = NNCreatorArgs()
expertArgs.inputs = [priv_obs]
# criticArgs.hidden_dim = [128, 64]
expertArgs.hidden_dim = [32]
expertArgs.outputs = [latent_space_size]

actor_std_noise = 1.

actorCritic = ActorCritic(actorArgs, criticArgs, actor_std_noise, expertArgs, debug_mess=True)
ppo = PPO(actorCritic, device=device, verbose=True)

reward_obj = Rewards(rollouts, device, reward_list, 0.999999, step_env, discrete_rewards=True)
pibb = PIBB(rollouts, h, 1, n_kernels * n_out, decay, variance, device="cuda:0", boost_noise=noise_boost)
env_config = EnvConfig()
config_env()

logger = Logger(save=SAVE_DATA, frequency=100, PIBB_param=pibb.get_hyper_parameters(), nn_config=config)
config_camera(ACTIVE_RECORDING_CAMERA, env_config, logger, step_env, int(1/0.01))

terrain_obj, terrain_curr = config_terrain(env_config)
alg_curr = config_learning_curriculum()
rad_curr = config_randomization_curriculum()
curricula = Curriculum(rollouts, device=device, terrain_config=terrain_curr, algorithm_config=alg_curr, randomization_config=rad_curr)
logged = False

policy = MLP_CPG(actorCritic, cpg_rbf_nn)

learning_algorithm = PPO_PIBB(ppo, pibb, curricula)

if RECOVER_CPG:
    learning_algorithm.read_data_point(cpg_filename, logger, policy, recover_MLP=False)


robot = Runner(policy, learning_algorithm, logger, config_file, env_config, reward_obj, n_out if not CURRICULUM_CPG_RBFN else 12,
               terrain_obj, curricula=curricula, verbose=True, store_observations=True)

try:
    robot.learn(max_iterations, step_env)
    logged = True
    logger.log(SAVE_DATA, show_final_graph, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
except KeyboardInterrupt:
    if not logged:
        logger.log(SAVE_DATA, True, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
