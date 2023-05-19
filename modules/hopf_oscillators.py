import math

import torch

# modular network
from modules.torchNet import torchNet
import matplotlib.pyplot as plt


class HopfOscillators(torchNet):
    def __init__(self, init_amplitude, init_phase,
                 intrinsic_frequency, intrinsic_amplitude,
                 command_signal_a, command_signal_d, expected_dt=None, maximum_vel=10., device=None, environments=1):
        super().__init__(device)
        self.num_envs = environments
        self.device = torch.device("cpu") if device is None else torch.device(device)

        self.amplitude = torch.zeros(self.num_envs, dtype=torch.float,
                                     device=self.device, requires_grad=False).fill_(init_amplitude)
        self.phase = torch.zeros(self.num_envs, dtype=torch.float,
                                 device=self.device, requires_grad=False).fill_(init_phase)
        self.expected_dt = expected_dt

        self.init_amplitude = init_amplitude
        self.init_phase = init_phase

        self.intrinsic_frequency = intrinsic_frequency
        self.intrinsic_amplitude = intrinsic_amplitude

        self.change_amplitude = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.change_phase = 0.0
        self.change_velocity_amplitude = 0.0

        self.command_signal_a = command_signal_a
        self.command_signal_d = command_signal_d

        self.maximum_command_phase = 1
        self.minimum_command_phase = -1

        self.maximum_phase_change = 0.
        self.stop_phase_change = -6.2832
        self.minimum_phase_change = -12.

        self.positive_range_change_phase = self.maximum_phase_change - self.stop_phase_change
        self.negative_range_change_phase = self.stop_phase_change - self.minimum_phase_change

        self.output = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def change_maximum_phase_change(self, maximum):
        if maximum < 0:
            raise ValueError("maximum phase change must be positive")

        if maximum > 12:
            maximum = 12.

        self.maximum_phase_change = maximum
        self.positive_range_change_phase = self.maximum_phase_change - self.stop_phase_change

        return maximum

    def change_default_dt(self, dt):
        self.expected_dt = dt

    def get_cpg_dt(self):
        return self.expected_dt

    def convert_phase_command(self, command):

        filter_ = torch.greater_equal(command, -0.005)
        filter_ &= torch.less_equal(command, 0.005)

        if len(filter_.nonzero(as_tuple=True)[0]):
            command[filter_] = 0.0

        command = command.clamp(min=self.minimum_command_phase, max=self.maximum_command_phase)

        positive = torch.greater_equal(command, 0.0)
        negative = torch.less(command, 0.0)

        if len(positive.nonzero(as_tuple=True)[0]):
            command[positive] = self.stop_phase_change + self.positive_range_change_phase * command[positive]

        if len(negative.nonzero(as_tuple=True)[0]):
            command[negative] = self.minimum_phase_change + self.negative_range_change_phase * \
                                (command[negative] - self.minimum_command_phase)


        return command * self.intrinsic_frequency

    def forward(self, modification_amplitude=None, phase_shift=None, dt=None):
        if dt is None:
            dt = self.expected_dt

        if phase_shift is None:
            phase_shift = self.ones

        if modification_amplitude is None:
            modification_amplitude = self.zeros

        phase_shift = self.convert_phase_command(phase_shift)

        self._get_new_phase(dt, phase_shift)
        self._get_new_amplitude(dt, modification_amplitude)
        self.output[:, 0] = self.amplitude * torch.sin(self.phase)
        self.output[:, 1] = self.amplitude * torch.cos(self.phase)

        return self.output

    def _get_new_amplitude(self, dt, modification_amplitude):
        prev_amplitude_change = self.change_amplitude
        prev_velocity_amplitude_change = self.change_velocity_amplitude

        aux = self.intrinsic_amplitude * self.command_signal_d - self.amplitude
        aux = aux*self.command_signal_a/4 - prev_velocity_amplitude_change
        self.change_velocity_amplitude = self.command_signal_a * aux + modification_amplitude

        self.change_amplitude += (self.change_velocity_amplitude + prev_velocity_amplitude_change) * dt / 2
        self.amplitude += (self.change_amplitude + prev_amplitude_change) * dt / 2

    def reset(self):
        self.amplitude = self.init_amplitude
        self.phase = self.init_phase

        self.change_amplitude = 0.0
        self.change_phase = 0.0
        self.change_velocity_amplitude = 0.0

    def _get_new_phase(self, dt, phase_shift):
        prev_amplitude_change = self.change_phase

        self.change_phase = 2 * math.pi * self.intrinsic_frequency * self.command_signal_d + phase_shift
        self.phase += (self.change_phase + prev_amplitude_change) * dt / 2


if __name__ == "__main__":
    init_amplitude = 0.2
    init_phase = 0.
    intrinsic_frequency = 1.0
    intrinsic_amplitude = 0.2
    command_signal_a = 1.0
    command_signal_d = 1
    expected_dt = 0.005 * 4

    cpg = HopfOscillators(init_amplitude, init_phase, intrinsic_frequency, intrinsic_amplitude, command_signal_a,
                          command_signal_d, expected_dt=expected_dt, environments=3)

    cpg_out_1 = []
    cpg_out_2 = []
    time = []
    # dt = 0.005 * 4
    phase = torch.Tensor([1, -1, 0.5])

    for i in range(200):

        x = cpg.forward(0.0, phase)
        x1 = x[:, 0].tolist()
        x2 = x[:, 1].tolist()
        time.append(i*expected_dt)
        cpg_out_1.append(x1)
        cpg_out_2.append(x2)
        print(x1)

    plt.style.use('seaborn-whitegrid')
    plt.plot(time, cpg_out_1, 'o')
    plt.plot(time, cpg_out_2, 'x')
    plt.show()