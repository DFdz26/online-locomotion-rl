class TestConfig:
    actions = None
    scale_actions = 0
    scale_hip = 0
    p_gain = None
    d_gain = None
    height = 0.
    deactivate_gravity = True

    def __init__(self) -> None:
        pass


class Observations:
    enable_observe_friction = True
    enable_observe_restitution= True
    enable_observe_motor_strength = True
    enable_observe_payload = True

    class NoiseObservation:
#         Noise for gyroscope = std dev 0.02
# for imu std dev 0.003
# for position std dev 0.006
# for velocity std dev 0.02
        enable_noise = False
        noise = [
            0.003,
            0.006,
            0.02,
            0.02,
            0.003
        ]


class Sensors:
    class Activations:
        height_measurement_activated = False
        camera_activated = False

    class HeightMeasurement:
        # Default config in x: 10-50cm on each side
        # Default config in y: 20-80cm on each side
        x_mesh = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
        y_mesh = [-4, -2, 2, 4]

        x_scale = 0.1
        y_scale = 0.1

    class Camera:
        n_camera = 1
        height = 480
        width = 640

    


class EnvConfig:
    def __init__(self):
        self.default_joints_angles = None
        self.rollout_time = 3
        self.iterations_without_control = 0
        self.num_env = 1
        self.spacing_env = -2
        self.spacing_env_x = 2
        self.dt = 0.005
        self.disable_leg = False
        self.num_env_colums = 10

        self.clip_actions = 100.0
        self.clip_observations = 100.0
        self.render_GUI = True

        self.actions_scale = 0.2
        self.hip_scale = 0.5

        self.test_joints = False
        self.joint_to_test = 1

        self.position_control = True

        self.test_config = TestConfig()
        self.sensors = Sensors()

        self.cfg_observations = Observations()
        