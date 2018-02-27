import numpy as np
from matplotlib import pyplot as plt


class Module:
    def __init__(self, size, s_pairs, learning_rate, tau, noise_max_range):
        self.size = size
        self.s_pairs = s_pairs
        self.tau_head = tau
        self.tau_fast = tau / 20

        self.head = np.zeros(size)
        self.head_out = [np.zeros(size)]
        self.should_input = np.zeros((self.s_pairs, size))
        self.should = np.zeros((self.s_pairs, size))
        self.should_out = [np.zeros((self.s_pairs, size))]
        self.should_not_out = [np.zeros((self.s_pairs, size))]
        self.inhibition = 0

        self.to_s_pairs = []
        self.from_modules = [None]*self.s_pairs
        self.weights = [None]*self.s_pairs
        self.weights_fitness = [None]*self.s_pairs

        self.learning_rate = learning_rate

        self.noise = 0
        self.noise_target = 0
        self.noise_step = 0
        self.noise_period = tau
        self.noise_alpha = 0.9
        self.noise_max_range = noise_max_range
        self.noise_range = noise_max_range*np.ones(self.size)

    def slow_noise(self):
        self.noise_range = np.clip(self.noise_range + 0.0000002 - 0.0002*self.head_out[-1], 0, self.noise_max_range)
        if self.noise_step < self.noise_period:
            self.noise_step += 1
        else:
            self.noise_step = 0
            self.noise_target = np.random.uniform(-self.noise_range, self.noise_range, self.size)
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
            self.weights_fitness[s_pair] = np.zeros((input_size, self.size))

    def step(self, head_input, s_input):
        # calculate input to 'should' neurons and update weights
        for s_pair in self.to_s_pairs:
            input_values = []
            for module in self.from_modules[s_pair]:
                input_values = np.append(input_values, module.head_out[-1])
            self.should_input[s_pair] = np.dot(input_values, self.weights[s_pair])
            self.weights[s_pair] += self.learning_rate*np.dot(input_values[np.newaxis].transpose(),
                                                              -self.should[s_pair][np.newaxis])
            # self.weights[s_pair] = self.weights[s_pair]/np.maximum(np.sum(self.weights[s_pair], 0), 1)

        # update the activity of neurons
        self.head += (-self.head + 2*self.head_out[-1] - self.inhibition + head_input + self.should_out[-1][0]
                      - self.should_not_out[-1][0] + self.slow_noise()) / self.tau_head
        self.head_out.append(np.clip(np.tanh(3*self.head), 0, 1))
        self.inhibition += (-self.inhibition + np.sum(self.head_out[-1])) / self.tau_fast

        self.should += (-self.should - self.head_out[-1] + s_input + self.should_input) / self.tau_fast
        self.should_out.append(np.clip(self.should, 0, 1))
        self.should_not_out.append(np.clip(-self.should, 0, 1))

    def plot_heads(self):
        fig, ax = plt.subplots()
        for head_num in range(self.size):
            ax.plot(np.array(self.head_out)[:, head_num], label=str(head_num))
        plt.legend()
        plt.show()

    def plot_circuits(self, show):
        fig, ax = plt.subplots(self.size, self.s_pairs, sharex=True, sharey=True)
        for circuit_num in range(self.size):
            for s_pair_num in range(self.s_pairs):
                axes_indices = self.axes_indices(circuit_num, s_pair_num)
                ax[axes_indices].plot(np.array(self.head_out)[:, circuit_num], 'b',
                                                 label=r"$H_" + str(circuit_num) + "$")
                ax[axes_indices].plot(np.array(self.should_not_out)[:, s_pair_num, circuit_num], 'r',
                                                 label=r"$N_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")
                ax[axes_indices].plot(np.array(self.should_out)[:, s_pair_num, circuit_num], 'g',
                                                 label=r"$S_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")

                ax[self.axes_indices(circuit_num, s_pair_num)].legend(loc='lower right')
        if show:
            plt.show()

    def axes_indices(self, circuit_num, s_pair_num):
        if self.s_pairs == 1 or self.size == 1:
            return max(circuit_num, s_pair_num)
        else:
            return circuit_num, s_pair_num

