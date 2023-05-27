from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch
import math

from learningAlgorithm.PIBB.PIBB import PIBB
from isaacGymConfig.envConfig import EnvConfig
# from isaacGymConfig.RobotConfig import RobotConfig
from Runner import Runner
from learningAlgorithm.PPO.History import History

from modules.cpgrbfn2 import CPGRBFN
from modules.logger import Logger

from isaacGymConfig.Rewards import Rewards
from learningAlgorithm.PPO.ActorCritic import ActorCritic
from learningAlgorithm.PPO.ActorCritic import NNCreatorArgs
from learningAlgorithm.PPO.PPO import PPO
from learningAlgorithm.PPO.PPO import PPOArgs

from learningAlgorithm.PIBB.PIBB import PIBB

from learningAlgorithm.CPG_MLP import MLP_CPG
from learningAlgorithm.CPG_MLP import PPO_PIBB
from isaacGymConfig.TerrainConfig import Terrain, TerrainComCfg
from isaacGymConfig.Curriculum import Curriculum, TerrainCurrCfg, AlgorithmCurrCfg, RandomizationCurrCfg

# config_file = "models/configs/config_minicheeta.json"
# graph_name = "graph_minicheeta_learning"
# cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/mini_cheetah/06_04_2023__22_07_23/300.pickle"
cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/b1/2023_05_21.09_58_06/150.pickle"
cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/b1/2023_05_22.18_08_22/150.pickle"
cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/b1/2023_05_22.23_04_36/150.pickle"
cpg_filename = "/home/danny/Downloads/online-locomotion-rl/runs/b1/2023_05_23.21_45_03/150.pickle"

config_file = "models/configs/config_b1.json"
graph_name = "graph_b1_learning"

ACTIVE_RECORDING_CAMERA = True
ACTIVATE_HEIGHT_READ = True
ACTIVATE_CPG_LATENT_HEAD = True
ACTIVATE_PHI_AMPLITUDE_HEAD = False
frequency_recording = 100
frequency_logger = 100 
frequency_plot = 2

CURRICULUM_CPG_RBFN = True
RENDER_GUI = False
SAVE_DATA = True
RECOVER_CPG = True
LOAD_CACHE = True
TERRAIN_CURRICULUM = True
ACTIVATE_HISTORY = True
# rollouts = 1500
rollouts = 1250
iterations_without_control = 4
num_env_colums = 100
# learning_rate_PPO = 0.0000003  # 0.0000003
start_PPO_acting_iteration = 150
num_mini_batches = 2100
num_mini_batches = int(150 * 2 * 2 / 4) #/4
num_mini_batches = int(150 * 2 * 2 / 6) #/4
num_mini_batches = int(4) #/4
num_prev_obs = 15
device = "cuda:0"
show_PPO_graph = True

intrinsic_frequency_cpg = 7.5
changing_direct_iteration = 41
changing_direct_iteration = 100 # 100
max_min_actor = 1
start_influence_PPO = 0.5
delay_terrains = 150

if RECOVER_CPG:
    start_PPO_acting_iteration = 0


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
    algCfg.PPOCfg.n_iterations_learning_from_CPG_RBFN = 999
    algCfg.PPOCfg.divider_initial_steps = 1.
    algCfg.PPOCfg.boost_kl_distance = 45.
    algCfg.PPOCfg.decay_boost_kl_distance = 0.92
    algCfg.PPOCfg.start_with_influence = start_influence_PPO
    algCfg.PPOCfg.change_rw_iter = delay_terrains
    algCfg.PPOCfg.step_increase_rw = 0.025
    algCfg.PPOCfg.gamma_filter = 0.1

    if CURRICULUM_CPG_RBFN and not RECOVER_CPG:
        algCfg.PIBBCfg.switching_indirect_to_direct = True
        algCfg.PIBBCfg.threshold_switching = changing_direct_iteration
        algCfg.PIBBCfg.decay_at_switching = 0.995
        algCfg.PIBBCfg.variance_at_switching = 0.009
        algCfg.PIBBCfg.boost_first_switching_noise = 1.
        algCfg.PIBBCfg.change_RW_scales_when_switching = True

    return algCfg


def config_randomization_curriculum():
    randCurrCfg = RandomizationCurrCfg()

    randCurrCfg.MotorParameters.randomize_motor_strength = True
    randCurrCfg.MotorParameters.step_randomization_motor = 0.

    randCurrCfg.ModelParameters.randomize_payload = True
    randCurrCfg.ModelParameters.payload_range = [[-1, -1], [1, 8]]
    randCurrCfg.ModelParameters.step_randomization_payload = 0.

    randCurrCfg.ModelParameters.randomize_friction = True
    randCurrCfg.ModelParameters.step_randomization_friction = 0.

    randCurrCfg.ModelParameters.randomize_restitution = True
    randCurrCfg.ModelParameters.step_randomization_restitution = 0.

    randCurrCfg.FrequencyControl.randomization_range = [iterations_without_control - 1,
                                                        iterations_without_control + 1]
    randCurrCfg.FrequencyControl.randomize_frquency_control = False

    randCurrCfg.Control.randomization_activated = True
    randCurrCfg.Control.generate_first_randomization = True
    randCurrCfg.Control.start_randomization_iteration = start_PPO_acting_iteration
    randCurrCfg.Control.randomization_interval_iterations = 3
    randCurrCfg.Control.randomization_frequency_iteration = 4
    randCurrCfg.Control.start_randomization_frequency_iteration = changing_direct_iteration

    return randCurrCfg


def config_terrain(env_config):
    list_terrains = [
        {
            "terrain": "flat_terrain",
        },
        # {
        #     "terrain": "random_uniform_terrain",
        #     "min_height": -0.010,
        #     "max_height": 0.010,
        #     "step": 0.010,
        #     "downsampled_scale": 0.5
        # },
        # {
        #     "terrain": "random_uniform_terrain",
        #     "min_height": -0.035,
        #     "max_height": 0.035,
        #     "step": 0.035,
        #     "downsampled_scale": 0.5
        # },
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
        },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.15,
            "max_height": 0.15,
            "step": 0.025,
            "downsampled_scale": 0.5
        },
        # # 
        # #     "terrain": "wave_terrain",
        # #     "num_waves": 2.,
        # #     "amplitude": 0.4,
        # #     "downsampled_scale": 0.5
        # # },
        # {
        #     "terrain": "stairs_terrain",
        #     "step_width": 0.5,
        #     "step_height": 0.15,
        #     "downsampled_scale": 0.5
        # },
        # {
        #     "terrain": "random_uniform_terrain",
        #     "min_height": -0.3,
        #     "max_height": 0.3,
        #     "step": 0.025,
        #     "downsampled_scale": 0.5
        # },
        {
            "terrain": "random_uniform_terrain",
            "min_height": -0.3,
            "max_height": 0.3,
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
    terrain_com_conf.terrain_width = y
    terrain_obj = Terrain(device, rollouts, list_terrains, terrain_com_conf)

    if TERRAIN_CURRICULUM:
        curriculum_terr = TerrainCurrCfg()

        curriculum_terr.object = terrain_obj
        delay = delay_terrains + 99999999
        start = 20
        first_curr = start_PPO_acting_iteration + delay_terrains
        # first_curr = start_PPO_acting_iteration + start + delay -20 -10 - 10 + 10
        second_curr = start_PPO_acting_iteration + start + 50 + delay - 40 -10 - 20 + 20 + 10
        third_curr = start_PPO_acting_iteration + start + 60 + delay -60 -10 - 30 + 30 + 80 + 20
        fourth_curr = start_PPO_acting_iteration + start + 70 + 60 + delay - 80 - 10 - 40 + 4 + 120 + 40# 700 
        fifth_curr = start_PPO_acting_iteration + 360 + delay -100  -10 - 50 + 50 + 150 + 100# 1750
        six_curr = 1750

        curriculum_terr.Control.threshold = {
            first_curr: False,
            second_curr: False,
            third_curr: False,
            fourth_curr: False,
            fifth_curr: False,
            six_curr: False,}
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

    env_config.sensors.Activations.height_measurement_activated = ACTIVATE_HEIGHT_READ

    env_config.cfg_observations.enable_observe_friction = False
    env_config.cfg_observations.enable_observe_restitution = False
    env_config.cfg_observations.enable_observe_motor_strength = True
    env_config.cfg_observations.enable_observe_payload = True
    env_config.cfg_observations.NoiseObservation.enable_noise = True
    


reward_list = {
    "x_distance": {
        "weight": 2.2 * 1 * 1.1,
        "reward_data": {
            "absolute_distance": False
        }
    },

    "y_distance": {
        "weight": -0.06 * 3. * 1.2 * 3,
        "reward_data": {
            "absolute_distance": True
        }
    },

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
        "weight": -0.25 * 1.5 * 1.15 * 1.8,
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
        "weight": -2.9 * 1.2 * 6 * 3 * 12 * 1.5,
        "reward_data": {
            "max_clip": 2.5,
        }
    },

    "low_penalization_contacts": {
        "weight": -0.25 * 1.5,
        "reward_data": {
            "absolute_distance": True,
            "max_clip": 2.5,
            "weights": {
                "correction_state": 0.02,
                "distance": 0.5,
            }
        }
    },

    "slippery": {
        "weight": 0.3,
        "reward_data": {
            # "slippery_coef": -0.0088,
            "slippery_coef": -0.9,
        }
    },

    # "smoothness": {
    #     "weight": 1.,
    #     "reward_data": {
    #         "jerk_coef": -0.00000008,
    #     }
    # },

    "z_vel": {
        "weight": 0.1 * 10. * 4.2 * 1.5 * 1.2,
        "reward_data": {
            "exponential": False,
            "weight": -0.24
        }
    },

    "roll_pitch": {
        "weight": 0.077 * 1.2 * 2.7 * 2.5 * 4 * 1.5,
        "reward_data": {
            "exponential": False,
            "weight": -0.15
        }
    },

    "yaw_vel": {
        "weight": 0.028 * 1.1 * 1.5 * 1.5 * 1,
        "reward_data": {
            "exponential": False,
            "weight": -0.1,
            "command": 0.,
        }
    },

    "y_velocity": {
        "weight": 0.1 * 1.6 * 3,
        "reward_data": {
            "exponential": False,
            "weight": -0.075  # 0.05
        }
    },

    "x_velocity": {
        "weight": 1. * 2 * 1.8 * 2.1 * 1.7 * 1.5,  # 3.8
        "reward_data": {
            "exponential": False,
            "weight": 0.178  # 0.177
        }
    },

    "velocity_smoothness": {
        "weight": 0.08 * 0.75,
        "reward_data": {
            "weight_vel": 0.01,
            "weight_acc": 0.00002,
            "weight": -0.0005,
        }
    },

    "limits": {
        "weight": 0.1,
        "reward_data": {
            "velocity_limits": 1.,
            "joint_limits": 1.,
            "weight": -1,
        }
    },

    "ppo_penalization": {
        "weight": -10.,
        "discount_level": 0.25
    }

    # "vel_cont": {
    #     "weight": -0.2,  # 0.25
    # },

    # "orthogonal_angle_error": {
    #     "weight": 0.1,
    #     "reward_data": {
    #         "weight": -0.02
    #     }
    # },

    # "changed_actions": {
    #     "weight": 2. * 0.07,
    #     "reward_data": {
    #         "weight": 0.1,
    #     }
    # },

}

reward_list = {
    "x_distance": {
    "weight": 2.2,
    "reward_data": {
      "absolute_distance": False
    }
  },
  "y_distance": {
    "weight": -0.18 * 2 * 1.4 * 2,
    "reward_data": {
      "absolute_distance": True
    }
  },
  "high_penalization_contacts": {
    "weight": -2.5 * 10 * 1.7 * 3 * 3,
    "reward_data": {
      "absolute_distance": True,
      "max_clip": 2.5,
      "weights": {
        "correction_state": 0.02,
        "distance": 0.5
      }
    }
  },
  "height_error": {
    "weight": -23.88,
    "reward_data": {
      "max_clip": 2.5
    }
  },
  "slippery": {
    "weight": 0.3 * 10 * 5 * 1.2 * 2,
    "reward_data": {
      "slippery_coef": -0.9 * 1.5*1.2
    }
  },
  "z_vel": {
    "weight": 4.2 * 5,
    "reward_data": {
      "exponential": False,
      "weight": -0.24
    }
  },
  "roll_pitch": {
    "weight": 0.24948 * 15 * 2 * 1.5 * 1.5, # for 1.5 cph, * 1.5 
    "reward_data": {
      "exponential": False,
      "weight": -0.15
    }
  },
  "torque_penalization": {
      "weight": 0.2, # for 1.5 cph, * 1.5 
        "reward_data": {
        "weight": -0.0005
        }
  },
  "yaw_vel": {
    "weight": 0.046200000000000005 * 1.2,
    "reward_data": {
      "exponential": False,
      "weight": -0.1,
      "command": 0.0
    }
  },
  "y_velocity": {
    "weight": 0.16000000000000003 * 1.5 * 6,
    "reward_data": {
      "exponential": False,
      "weight": -0.075
    }
  },
  "x_velocity": {
    "weight": 9.072000000000001 * 2 * 5 * 2,
    "reward_data": {
      "exponential": False,
      "weight": 0.178
    }
  },
  "velocity_smoothness": {
    "weight": 0.06 * 10 * 1.5 * 2,
    "reward_data": {
      "weight_vel": 0.01,
      "weight_acc": 2e-05,
      "weight": -0.00005
    }
  },
  "limits": {
    "weight": 0.1,
    "reward_data": {
      "velocity_limits": 1.0,
      "joint_limits": 1.0,
      "weight": -1
    }
  },

  "noise_ppo_penalization": {
        "weight": -0,
        "discount_level": 0.25,
    },


    "ppo_penalization": {
        "weight": -50. * 20 * 10 * 50,
        "discount_level": 0.25,
    },
    "low_penalization_contacts": {
        "weight": -0.25 * 1.5,
        "reward_data": {
            "absolute_distance": True,
            "max_clip": 2.5,
            "weights": {
                "correction_state": 0.02,
                "distance": 0.5,
            }
        }
    },
    "stand_phase_error": {
            "weight": 0,
            "reward_data": {
                "absolute_distance": True,
                "max_clip": 2.5,
            }
            }
}

n_kernels = 20
h = 10

if RECOVER_CPG:
    decay = 0.992
    variance = 0.015
    variance = 0.0
    noise_boost = 0.1
    noise_boost = 0.0
elif CURRICULUM_CPG_RBFN:
    decay = 0.992
    variance = 0.027
    # variance = 0.009
    # variance = 0.021
    noise_boost = 1.5
else:
    variance = 0.027
    decay = 0.9965
    noise_boost = 1.75

dt = 0.005
seconds_iteration = 5 / 2
max_iterations = 99001
step_env = 200
# step_env = int(seconds_iteration / 0.01)

show_final_graph = True

if CURRICULUM_CPG_RBFN:
    encoding = "indirect"
else:
    # encoding = "indirect"
    encoding = "direct"

if RECOVER_CPG:
    encoding = "direct"

# actions_scale = 0.2
actions_scale = 0.25
# hip_scale = 0.2
hip_scale = 0.12

hyperparam = {
    "NIN": 1,
    "NSTATE": n_kernels,
    "MOTORS_LEG": 3,
    "NLEG": 4,
    "TINIT": 1000,
    "ENCODING": encoding,
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
        "INTRINSIC_FREQUENCY": intrinsic_frequency_cpg,
        "INTRINSIC_AMPLITUDE": 0.2,
        "COMMAND_SIGNAL_A": 1.0,
        "COMMAND_SIGNAL_D": 1.0,
        "EXPECTED_DT": dt * iterations_without_control/10,
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
    "device": "cuda",
    "HYPERPARAM": hyperparam,
    "RBF": rbf_param,
    "CPG": cpg_param,
    "UTILS": cpg_utils,
    "deactivate_hip": False
}

cpg_rbf_nn = CPGRBFN(config, dimensions=rollouts, load_cache=LOAD_CACHE)

n_out = cpg_rbf_nn.get_n_outputs()
print(f"n_out: {n_out}")

latent_space_size = 12 + 8
latent_heads_size = 32
priv_obs = 21 + 3
head_cpg_latent = None
head_phi_amplitude = None

if ACTIVATE_HEIGHT_READ:
    priv_obs += 52

# n_observations = 45
n_observations = 42 + 1
actor_input = n_observations + latent_space_size

actorArgs = NNCreatorArgs()
# actorArgs.inputs = [39]
actorArgs.inputs = [actor_input]
# actorArgs.hidden_dim = [128, 64]
# actorArgs.hidden_dim = [256, 128]
actorArgs.hidden_dim = [256, 128]
actorArgs.outputs = [n_out if not CURRICULUM_CPG_RBFN else 12]

criticArgs = NNCreatorArgs()
criticArgs.inputs = [actor_input]
# criticArgs.hidden_dim = [128, 64]
criticArgs.hidden_dim = [256, 128]
criticArgs.outputs = [1]

expertArgs = NNCreatorArgs()
expertArgs.inputs = [priv_obs]
# criticArgs.hidden_dim = [128, 64]
if ACTIVATE_HEIGHT_READ:
    expertArgs.hidden_dim = [128, 64]
else:
    expertArgs.hidden_dim = [64]
expertArgs.hidden_dim = [256, 128]
if ACTIVATE_CPG_LATENT_HEAD or ACTIVATE_PHI_AMPLITUDE_HEAD:
    expertArgs.outputs = [latent_heads_size]

    if ACTIVATE_CPG_LATENT_HEAD:
        head_cpg_latent = NNCreatorArgs()
        head_cpg_latent.inputs = [latent_heads_size + 12]
        head_cpg_latent.hidden_dim = [32]
        head_cpg_latent.outputs = [latent_space_size]

    if ACTIVATE_PHI_AMPLITUDE_HEAD:
        head_phi_amplitude = NNCreatorArgs()
        head_phi_amplitude.inputs = [latent_heads_size]
        head_phi_amplitude.hidden_dim = [32]
        head_phi_amplitude.outputs = [1]

    space_student = latent_heads_size
else:
    expertArgs.outputs = [latent_space_size]
    space_student = latent_space_size

studentArgs = NNCreatorArgs()
studentArgs.inputs = [num_prev_obs * n_observations]
# criticArgs.hidden_dim = [128, 64]
studentArgs.hidden_dim = [256, 64]
studentArgs.outputs = [space_student]

actor_std_noise = 1.

actorCritic = ActorCritic(actorArgs, criticArgs, actor_std_noise, expertArgs, studentArgs, debug_mess=True,
                          scale_max=max_min_actor, scale_min=-max_min_actor, head_encoder_cpg_actions=head_cpg_latent,
                          head_cpg_phi_amplitude=head_phi_amplitude)

ppo_cfg = PPOArgs()
ppo_cfg.num_past_actions = num_prev_obs
ppo_cfg.num_mini_batches = num_mini_batches
ppo = PPO(actorCritic, device=device, verbose=True, cfg=ppo_cfg, store_primitive_movement=ACTIVATE_CPG_LATENT_HEAD)


pibb = PIBB(rollouts, h, 1, n_kernels * n_out, decay, variance, device="cuda:0", boost_noise=noise_boost)
env_config = EnvConfig()
config_env()

logger = Logger(save=SAVE_DATA, frequency=frequency_logger, frequency_plot=frequency_plot,
                PIBB_param=pibb.get_hyper_parameters(), nn_config=config, show_PPO_graph=show_PPO_graph)
reward_obj = Rewards(rollouts, device, reward_list, 0.999999, step_env, logger, discrete_rewards=True)
config_camera(ACTIVE_RECORDING_CAMERA, env_config, logger, step_env, int(1 / 0.01))

terrain_obj, terrain_curr = config_terrain(env_config)
alg_curr = config_learning_curriculum()
rad_curr = config_randomization_curriculum()
curricula = Curriculum(rollouts, device=device, terrain_config=terrain_curr, algorithm_config=alg_curr,
                       randomization_config=rad_curr)
logged = False

policy = MLP_CPG(actorCritic, cpg_rbf_nn)

learning_algorithm = PPO_PIBB(ppo, pibb, curricula)

if RECOVER_CPG:
    learning_algorithm.read_data_point(cpg_filename, logger, policy, recover_MLP=False)

history_obj = History(rollouts, num_prev_obs, observation_shape=n_observations,
                      device=device) if ACTIVATE_HISTORY else None
robot = Runner(policy, learning_algorithm, logger, config_file, env_config, reward_obj,
               n_out if not CURRICULUM_CPG_RBFN else 12,
               terrain_obj, curricula=curricula, verbose=True, store_observations=True, history_obj=history_obj, device=device)

try:
    robot.learn(max_iterations, step_env)
    logged = True
    logger.plot_log(SAVE_DATA, show_final_graph, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
except KeyboardInterrupt:
    if not logged:
        logger.plot_log(SAVE_DATA, True, plot_file_name=graph_name, save_datapoint=SAVE_DATA)
