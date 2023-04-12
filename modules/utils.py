"""
Class: Utils
created by: Daniel Fernandez
e-mail: dafer21@student.sdu.dk
date last modification: 22-01-23

"""

# ------------------- import modules ---------------------
import torch


# ------------------- class CPGUtils ---------------------

class CPGUtils:

    # ---------------------- constructor ------------------------
    def __init__(self, config, CPG, verbose=False):

        self.device = config["device"]
        self.config = config["UTILS"]
        self.CPG = CPG
        self.period = 0
        self.verbose = verbose

    def init_buffers(self):
        self._discover_CPG_period_()
        self._create_buffers_()

        for _ in range(self.period):
            self.store_cpg_steps(self.CPG())

    def read_stored(self, steps_before):
        index = (self.step - steps_before) % self.period

        return self.buffer[index].detach().clone()

    def _increase_step_(self):
        self.step += 1
        self.step %= self.period

    def store_cpg_steps(self, cpg_out):
        self.buffer[self.step] = torch.flatten(cpg_out.detach().clone())
        self._increase_step_()

    def _create_buffers_(self):
        self.buffer = torch.zeros(size=(self.period, 2), dtype=torch.float, device=self.device, requires_grad=False)
        self.step = 0

    def _discover_CPG_period_(self):

        steps_int = self.config["STEPS_INIT"]
        show_graphic = self.config["SHOW_GRAPHIC"]

        buff_CPG0 = []
        buff_CPG1 = []
        y = []

        started_buff = []
        period_buff = []

        samples = 0
        previous_signal = 0

        first_peak = True

        for i in range(steps_int):
            cpgoutput = self.CPG().detach().cpu().numpy()
            signal0 = cpgoutput[0]
            signal1 = cpgoutput[1]

            if samples == len(started_buff):
                started_buff.append(i)
            elif previous_signal > signal0 and first_peak:

                first_peak = False
            elif previous_signal < signal0 and not first_peak:

                period_buff.append(i - started_buff[samples])

                samples += 1
                first_peak = True

            previous_signal = signal0

            buff_CPG0.append(signal0)
            buff_CPG1.append(signal1)
            y.append(i)

        if samples < 3 or show_graphic:
            import matplotlib.pyplot as plt

            plt.plot(y, buff_CPG0, label='Output 0')
            plt.plot(y, buff_CPG1, label='Output 1')
            plt.legend()
            plt.xlabel("Steps")
            plt.show()

        if samples < 3:
            raise Exception(
                f"There are no enough samples for computing the frequency of the CPG, increase the initial "
                f"steps.\nCurrent: {steps_int}")

        _ = period_buff.pop(0)
        self.period = sum(period_buff) / (samples - 1)

        if self.verbose:
            print(f"Period CPG: {self.period}.")

        self.period = int(self.period)
