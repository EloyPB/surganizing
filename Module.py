import numpy as np
from matplotlib import pyplot as plt


class Module:
    def __init__(self, size, s_pairs, learning_rate, tau):
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

        self.learning_rate = learning_rate

        self.saturation = np.vectorize(saturation, otypes=[np.float])
        self.activation = np.vectorize(activation, otypes=[np.float])

    def enable_connections(self, to_s_pair, from_modules):
        self.to_s_pairs.append(to_s_pair)
        self.from_modules[to_s_pair] = from_modules

    def initialize_weights(self):
        for s_pair in self.to_s_pairs:
            input_size = 0
            for module in self.from_modules[s_pair]:
                input_size += module.size
            self.weights[s_pair] = np.zeros((input_size, self.size))

    def step(self, head_input, s_input):
        for s_pair in self.to_s_pairs:
            input_values = []
            for module in self.from_modules[s_pair]:
                input_values = np.append(input_values, module.head_out[-1])
            self.should_input[s_pair] = np.dot(input_values, self.weights[s_pair])
            self.weights[s_pair] += self.learning_rate*np.dot(input_values[np.newaxis].transpose(),
                                                              -self.should[s_pair][np.newaxis])

        self.head += (-self.head + 2*self.head_out[-1] - self.inhibition + head_input + self.should_out[-1][0]
                      - self.should_not_out[-1][0]) / self.tau_head
        self.head_out.append(self.saturation(np.tanh(3*self.head)))
        self.inhibition += (-self.inhibition + np.sum(self.head_out[-1])) / self.tau_fast

        self.should += (-self.should - self.head_out[-1] + s_input + self.should_input) / self.tau_fast
        self.should_out.append(self.saturation(self.should))
        self.should_not_out.append(self.saturation(-self.should))

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


def saturation(value):
    """Saturates 'value' between 0 and 1"""
    if value < 0:
        return 0
    elif value > 1:
        return 1
    else:
        return value


def activation(value):
    """Activation function for the head neurons"""
    if value <= 0.25:
        return 0
    elif value <= 0.75:
        return 2 * value - 0.5
    else:
        return 1

