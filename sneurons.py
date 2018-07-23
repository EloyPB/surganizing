import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class SNeurons:
    def __init__(self, shape, noise_samples, noise_amplitude, log_head=False, log_head_out=True):
        self.shape = tuple(shape)  # shape of the array of mismatch detection circuits

        self.head = np.zeros(shape)  # activity of head neurons
        self.log_head = log_head
        if log_head:
            self.head_log = [np.zeros(shape)]

        self.head_out = np.zeros(shape)  # activity of head neurons after activation function
        self.log_head_out = log_head_out
        if log_head_out:
            self.head_out_log = [np.zeros(shape)]
        self.head_out_k = 3  # constant for the activation function

        shape.insert(0, 3)
        self.noise = np.random.uniform(-noise_amplitude, noise_amplitude, shape)
        self.noise_amplitude = noise_amplitude
        self.noise_samples = noise_samples
        self.noise_sample_num = 0

    def slow_noise(self):
        alpha = self.noise_sample_num / self.noise_samples
        self.noise[1] = (1-alpha)*self.noise[0] + alpha*self.noise[2]
        self.noise_sample_num += 1

        if self.noise_sample_num == self.noise_samples:
            self.noise[0] = deepcopy(self.noise[2])
            self.noise[2] = np.random.uniform(-self.noise_amplitude, self.noise_amplitude, self.shape)
            self.noise_sample_num = 0

        return self.noise[1]

    def step(self):
        self.head = self.slow_noise()
        if self.log_head:
            self.head_log.append(deepcopy(self.head))

        self.head_out = np.clip(np.tanh(self.head_out_k * self.head), 0, 1)
        if self.log_head_out:
            self.head_out_log.append(deepcopy(self.head_out))

    def run(self, num_steps):
        for step_num in range(num_steps):
            self.step()

    def plot(self):
        plt.plot(np.array(self.head_out_log)[:,0,0], '*-')
        plt.show()
