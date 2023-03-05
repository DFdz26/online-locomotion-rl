from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from params_proto.proto import PrefixProto


class BaseConfiguration:
    def __init__(self, cfg, dt_) -> None:
        self.sim = None
        self.gym = gymapi.acquire_gym()

        # parse arguments
        args = gymutil.parse_arguments(description="Asset and Environment Information")

        # create simulation context
        sim_params = gymapi.SimParams()

        class sim(PrefixProto, cli=False):
            dt = dt_
            substeps = 1
            gravity = [0., 0., -9.81]  # [m/s^2]
            up_axis = 1  # 0 is y, 1 is z

            use_gpu_pipeline = cfg['sim_params']['use_gpu']

            class physx(PrefixProto, cli=False):
                num_threads = 10
                solver_type = 1  # 0: pgs, 1: tgs
                num_position_iterations = 4
                num_velocity_iterations = 0
                contact_offset = 0.01  # [m]
                rest_offset = 0.0  # [m]
                bounce_threshold_velocity = 0.5  # 0.5 [m/s]
                max_depenetration_velocity = 1.0
                max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
                default_buffer_size_multiplier = 5
                contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.up_axis = 1
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        sim_params.use_gpu_pipeline = cfg["sim_params"]["use_gpu"]
        sim_params = gymapi.SimParams()
        sim_pars = gymutil.parse_sim_config(vars(sim), sim_params)
        # sim_params = sim_pars
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()


    def __del__(self):

        if not(self.sim is None):
                
            # Cleanup the simulator
            self.gym.destroy_sim(self.sim)
        
