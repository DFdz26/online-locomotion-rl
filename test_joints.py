from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch

from tkinter import *
from tkinter import ttk
from isaacGymConfig.envConfig import EnvConfig
from isaacGymConfig.RobotConfig import RobotConfig

init_p = 150.0
init_d = 0.5

config_file = "configs/config_b1.json"
scale_w = 20.0
scale_d = 50.0
scale_p = .05

env_config = EnvConfig()
ROOT = Tk()

hip = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
hip.set(50)

knee = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
knee.set(50)

ankle = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
ankle.set(50)

hip_p = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
hip_p.set(50)

knee_p = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
knee_p.set(50)

ankle_p = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
ankle_p.set(50)

hip_d = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
hip_d.set(50)

knee_d = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
knee_d.set(50)

ankle_d = Scale(ROOT, from_=0, to=100, length=600,tickinterval=1, orient=HORIZONTAL)
ankle_d.set(50)

error_hip = Label(ROOT, text = "0")
error_knee = Label(ROOT, text = "0")
error_ankle = Label(ROOT, text = "0")

torque_hip = Label(ROOT, text = "0")
torque_knee = Label(ROOT, text = "0")
torque_ankle = Label(ROOT, text = "0")

hip_p_l = Label(ROOT, text = "0")
knee_p_l = Label(ROOT, text = "0")
ankle_p_l = Label(ROOT, text = "0")

hip_d_l = Label(ROOT, text = "0")
knee_d_l = Label(ROOT, text = "0")
ankle_d_l = Label(ROOT, text = "0")

desired_hip = Label(ROOT, text = "0")
desired_knee = Label(ROOT, text = "0")
desired_ankle = Label(ROOT, text = "0")

actual_hip = Label(ROOT, text = "0")
actual_knee = Label(ROOT, text = "0")
actual_ankle = Label(ROOT, text = "0")

hip.pack()
knee.pack()
ankle.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

error_hip.pack()
error_knee.pack()
error_ankle.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

torque_hip.pack()
torque_knee.pack()
torque_ankle.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

hip_p.pack()
hip_p_l.pack()

knee_p.pack()
knee_p_l.pack()

ankle_p.pack()
ankle_p_l.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

hip_d.pack()
hip_d_l.pack()

knee_d.pack()
knee_d_l.pack()

ankle_d.pack()
ankle_d_l.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

desired_hip.pack()
desired_knee.pack()
desired_ankle.pack()

separator = ttk.Separator(ROOT, orient='horizontal')
separator.pack(fill='x')

actual_hip.pack()
actual_knee.pack()
actual_ankle.pack()


def config_env ():
    env_config.num_env = 1
    env_config.test_joints = True
    env_config.joint_to_test = -1


def prep_test():
    env_config.test_config.actions = torch.zeros([1, 12], dtype=torch.float, device="cuda:0", requires_grad=False)
    env_config.test_config.p_gain = torch.zeros(12, dtype=torch.float, device="cuda:0", requires_grad=False)
    env_config.test_config.d_gain = torch.zeros(12, dtype=torch.float, device="cuda:0", requires_grad=False)

    env_config.test_config.p_gain.fill_(init_p)
    env_config.test_config.d_gain.fill_(init_d)

    env_config.test_config.height = 0.8

    env_config.test_config.scale_actions = 0.3
    env_config.test_config.scale_hip = 0.15


def continue_run(sim):
    while not sim.gym.query_viewer_has_closed(sim.viewer):
        ROOT.update()

        hip_value = hip.get()/scale_w - 50/scale_w
        knee_value = knee.get()/scale_w - 50/scale_w
        ankle_value = ankle.get()/scale_w - 50/scale_w

        hip_d_value = hip_d.get()/scale_d - 50/scale_d
        knee_d_value = knee_d.get()/scale_d - 50/scale_d
        ankle_d_value = ankle_d.get()/scale_d - 50/scale_d

        hip_p_value = hip_p.get()/scale_p - 50/scale_p
        knee_p_value = knee_p.get()/scale_p - 50/scale_p
        ankle_p_value = ankle_p.get()/scale_p - 50/scale_p

        env_config.test_config.actions[0, [0, 3, 6, 9]] = hip_value
        env_config.test_config.actions[0, [1, 4, 7, 10]] = knee_value
        env_config.test_config.actions[0, [2, 5, 8, 11]] = ankle_value        

        env_config.test_config.p_gain[[0, 3, 6, 9]] = init_p + hip_p_value
        env_config.test_config.p_gain[[1, 4, 7, 10]] = init_p +knee_p_value
        env_config.test_config.p_gain[[2, 5, 8, 11]] = init_p + ankle_p_value

        env_config.test_config.d_gain[[0, 3, 6, 9]] = init_d + hip_d_value
        env_config.test_config.d_gain[[1, 4, 7, 10]] = init_d +knee_d_value
        env_config.test_config.d_gain[[2, 5, 8, 11]] = init_d + ankle_d_value

        hip_d_l.config(text = f"{float(env_config.test_config.d_gain[0])}")
        knee_d_l.config(text = f"{float(env_config.test_config.d_gain[1])}")
        ankle_d_l.config(text = f"{float(env_config.test_config.d_gain[2])}")


        hip_p_l.config(text = f"{float(env_config.test_config.p_gain[0])}")
        knee_p_l.config(text = f"{float(env_config.test_config.p_gain[1])}")
        ankle_p_l.config(text = f"{float(env_config.test_config.p_gain[2])}")

        torque_hip.config(text = f"{float(sim.torques[0, 0])}")
        torque_knee.config(text = f"{float(sim.torques[0, 1])}")
        torque_ankle.config(text = f"{float(sim.torques[0, 2])}")

        sim.step(test_data=env_config.test_config, actions=None, position_control=True)

        error_hip.config(text = f"{float(sim.controller_error[0, 0])}")
        error_knee.config(text = f"{float(sim.controller_error[0, 1])}")
        error_ankle.config(text = f"{float(sim.controller_error[0, 2])}")

        torque_hip.config(text = f"{float(sim.torques[0, 0])}")
        torque_knee.config(text = f"{float(sim.torques[0, 1])}")
        torque_ankle.config(text = f"{float(sim.torques[0, 2])}")

        desired_hip.config(text = f"{float(sim.desired_config[0])}")
        desired_knee.config(text = f"{float(sim.desired_config[1])}")
        desired_ankle.config(text = f"{float(sim.desired_config[2])}")

        actual_hip.config(text = f"{float(sim.dof_pos[0, 0])}")
        actual_knee.config(text = f"{float(sim.dof_pos[0, 1])}")
        actual_ankle.config(text = f"{float(sim.dof_pos[0, 2])}")


prep_test()
config_env()

minicheeta = RobotConfig(config_file, env_config, None, None, None)
continue_run(minicheeta)


