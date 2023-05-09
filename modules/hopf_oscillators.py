import math

import torch

# modular network
from modules.torchNet import torchNet
import matplotlib.pyplot as plt


class HopfOscillators(torchNet):
    def __init__(self, init_amplitude, init_phase,
                 intrinsic_frequency, intrinsic_amplitude,
                 command_signal_a, command_signal_d, expected_dt=None, device=None):
        super().__init__(device)
        self.amplitude = init_amplitude
        self.phase = init_phase
        self.expected_dt = expected_dt
        self.device = torch.device("cpu") if device is None else torch.device(device)

        self.init_amplitude = init_amplitude
        self.init_phase = init_phase

        self.intrinsic_frequency = intrinsic_frequency
        self.intrinsic_amplitude = intrinsic_amplitude

        self.change_amplitude = 0.0
        self.change_phase = 0.0
        self.change_velocity_amplitude = 0.0

        self.command_signal_a = command_signal_a
        self.command_signal_d = command_signal_d

        self.output = torch.Tensor([0.0, 0.0]).to(device)

    def forward(self, modification_amplitude=0.0, phase_shift=0.0, dt=None):
        if dt is None:
            dt = self.expected_dt

        self._get_new_phase(dt, phase_shift)
        self._get_new_amplitude(dt, modification_amplitude)
        # print(self.amplitude * math.sin(self.phase))
        self.output = torch.Tensor([self.amplitude * math.cos(self.phase),
                                    self.amplitude * math.sin(self.phase)]).to(self.device)

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
                          command_signal_d, expected_dt=expected_dt)

    cpg_out_1 = []
    cpg_out_2 = []
    time = []
    # dt = 0.005 * 4
    phase = 0.0

    for i in range(200):

        if i == 100:
            phase = 12.
        elif i == 150:
            phase = 0.0

        x1, x2 = cpg.forward(0.0, phase)
        time.append(i*expected_dt)
        cpg_out_1.append(x1)
        cpg_out_2.append(x2)
        print(x1)

    plt.style.use('seaborn-whitegrid')
    plt.plot(time, cpg_out_1, 'o')
    plt.plot(time, cpg_out_2, 'x')
    plt.show()