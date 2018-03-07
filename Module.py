import numpy as np
from matplotlib import pyplot as plt


class Module:
    def __init__(self, size, s_pairs, s_pair_weights, learning_rate, time_constant, noise_max_amplitude,
                 log_h=False, log_h_out=True, log_s=False, log_s_out=True, log_sn_out=True, log_inhibition=False,
                 log_weights=True, log_noise_amplitude=False):

        self.size = size
        self.s_pairs = s_pairs
        self.s_pair_weights = np.array(s_pair_weights)
        self.learning_rate = learning_rate
        self.time_constant_h = time_constant
        self.fast_time_constant = time_constant / 20
        self.noise_max_amplitude = noise_max_amplitude
        self.log_h = log_h
        self.log_h_out = log_h_out
        self.log_s = log_s
        self.log_s_out = log_s_out
        self.log_sn_out = log_sn_out
        self.log_inhibition = log_inhibition
        self.log_weights = log_weights
        self.log_noise_amplitude = log_noise_amplitude

        self.h = np.zeros(size)
        if log_h:
            self.h_log = [np.zeros(size)]
        self.h_out = np.zeros(size)
        if log_h_out:
            self.h_out_log = [np.zeros(size)]
        self.s_input = np.zeros((self.s_pairs, size))
        self.s = np.zeros((self.s_pairs, size))
        if log_s:
            self.s_log = [np.zeros((self.s_pairs, size))]
        self.s_out = np.zeros((self.s_pairs, size))
        if log_s_out:
            self.s_out_log = [np.zeros((self.s_pairs, size))]
        self.sn_out = np.zeros((self.s_pairs, size))
        if log_sn_out:
            self.sn_out_log = [np.zeros((self.s_pairs, size))]
        self.inhibition = 0
        if log_inhibition:
            self.inhibition_log = [0]

        self.to_s_pairs = []
        self.from_modules = [None]*self.s_pairs
        self.weights = [None]*self.s_pairs
        if log_weights:
            self.weights_log = [[None]*self.s_pairs]

        self.noise = 0
        self.noise_target = 0
        self.noise_step = 0
        self.noise_period = time_constant
        self.noise_alpha = 0.9
        self.noise_amplitude = noise_max_amplitude*np.ones(self.size)
        if log_noise_amplitude:
            self.noise_amplitude_log = [noise_max_amplitude*np.ones(self.size)]
        
    def slow_noise(self):
        self.noise_amplitude = np.clip(self.noise_amplitude + 0.0000002 - 0.0002*self.h_out, 0,
                                       self.noise_max_amplitude)
        if self.log_noise_amplitude:
            self.noise_amplitude_log.append(self.noise_amplitude)

        if self.noise_step < self.noise_period:
            self.noise_step += 1
        else:
            self.noise_step = 0
            self.noise_target = np.random.uniform(-self.noise_amplitude, self.noise_amplitude, self.size)
        self.noise = self.noise_alpha*self.noise + (1-self.noise_alpha)*self.noise_target

        return self.noise

    def enable_connections(self, to_s_pair, from_modules):
        self.to_s_pairs.append(to_s_pair)
        self.from_modules[to_s_pair] = from_modules

    def initialize_weights(self):
        for s_pair in self.to_s_pairs:
            input_size = 0
            for module in self.from_modules[s_pair]:
                input_size += module.size
            self.weights[s_pair] = np.zeros((input_size, self.size))
            if self.log_weights:
                self.weights_log[0][s_pair] = np.zeros((input_size, self.size))

    def step(self, h_input, s_input):
        # calculate input to 's' neurons and update weights
        for s_pair in self.to_s_pairs:
            input_values = []
            for module in self.from_modules[s_pair]:
                input_values = np.append(input_values, module.h_out)
            self.s_input[s_pair] = np.dot(input_values, self.weights[s_pair])
            self.weights[s_pair] += self.learning_rate*np.dot(input_values[np.newaxis].transpose(),
                                                              -self.s[s_pair][np.newaxis])

        # update the activity of neurons
        self.h += (-self.h + 2*self.h_out - self.inhibition + h_input + np.dot(self.s_pair_weights, self.s_out)
                   - np.dot(self.s_pair_weights, self.sn_out) + self.slow_noise()) / self.time_constant_h
        if self.log_h:
            self.h_log.append(self.h)
        self.h_out = np.clip(np.tanh(3*self.h), 0, 1)
        if self.log_h_out:
            self.h_out_log.append(self.h_out)
        self.inhibition += (-self.inhibition + np.sum(self.h_out)) / self.fast_time_constant
        if self.log_inhibition:
            self.inhibition_log.append(self.log_inhibition)

        self.s += (-self.s - self.h_out + s_input + self.s_input) / self.fast_time_constant
        if self.log_s:
            self.s_log.append(self.s)
        self.s_out = np.clip(self.s, 0, 1)
        if self.log_s_out:
            self.s_out_log.append(self.s_out)
        self.sn_out = np.clip(-self.s, 0, 1)
        if self.log_sn_out:
            self.sn_out_log.append(self.sn_out)

    def plot_heads(self):
        fig, ax = plt.subplots()
        for h_num in range(self.size):
            ax.plot(np.array(self.h_out_log)[:, h_num], label=str(h_num))
        plt.legend()
        plt.show()

    def plot_circuits(self, show):
        fig, ax = plt.subplots(self.size, self.s_pairs, sharex=True, sharey=True)
        for circuit_num in range(self.size):
            for s_pair_num in range(self.s_pairs):
                axes_indices = self.axes_indices(circuit_num, s_pair_num)
                ax[axes_indices].plot(np.array(self.h_out_log)[:, circuit_num], 'b',
                                      label=r"$H_" + str(circuit_num) + "$")
                ax[axes_indices].plot(np.array(self.sn_out_log)[:, s_pair_num, circuit_num], 'r',
                                      label=r"$N_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")
                ax[axes_indices].plot(np.array(self.s_out_log)[:, s_pair_num, circuit_num], 'g',
                                      label=r"$S_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")

                ax[self.axes_indices(circuit_num, s_pair_num)].legend(loc='lower right')
        if show:
            plt.show()

    def axes_indices(self, circuit_num, s_pair_num):
        if self.s_pairs == 1 or self.size == 1:
            return max(circuit_num, s_pair_num)
        else:
            return circuit_num, s_pair_num

