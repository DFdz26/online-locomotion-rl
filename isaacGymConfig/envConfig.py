class test_config:
        actions = None
        scale_actions = 0
        scale_hip = 0
        p_gain = None
        d_gain = None
        height = 0.

        def __init__(self) -> None:
            pass


class EnvConfig:
    def __init__(self):
        self.default_joints_angles = None
        self.rollout_time = 3
        self.num_env = 1
        self.spacing_env = -3.5
        self.dt = 0.005
        self.disable_leg = False
        
        self.actions_scale = 0.2
        self.hip_scale = 0.5

        self.test_joints = False
        self.joint_to_test = 1

        self.position_control = True

        self.test_config = test_config()

    