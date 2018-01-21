import numpy as np
from matplotlib import pyplot as plt


class Module:
    def __init__(self, size, tau):
        self.size = size
        self.s_pairs = 2
        self.tau_head = tau
        self.tau_fast = tau / 20

        self.head = np.zeros(size)
        self.head_out = [np.zeros(size)]
        self.shouldnt = np.zeros((self.s_pairs, size))
        self.shouldnt_out = [np.zeros((self.s_pairs, size))]
        self.should_out = [np.zeros((self.s_pairs, size))]
        self.inhibition = 0

        self.child = None
        self.parent = None

        self.saturation = np.vectorize(saturation, otypes=[np.float])
        self.activation = np.vectorize(activation, otypes=[np.float])

    def is_child_of(self, parent):
        assert isinstance(parent, Module)
        self.parent = parent
        parent.child = self

    def step(self, predictions, head_input):
        self.head += (-self.head + 2*self.head_out[-1] - self.inhibition + head_input + self.should_out[-1][0]
                      - self.shouldnt_out[-1][0]) / self.tau_head
        self.head_out.append(self.saturation(np.tanh(3*self.head)))
        self.shouldnt += (-self.shouldnt + self.head_out[-1] - predictions) / self.tau_fast
        self.shouldnt_out.append(self.saturation(self.shouldnt))
        self.should_out.append(self.saturation(-self.shouldnt))
        self.inhibition += (-self.inhibition + np.sum(self.head_out[-1])) / self.tau_fast

    def plot_heads(self):
        fig, ax = plt.subplots()
        for head_num in range(self.size):
            ax.plot(np.array(self.head_out)[:, head_num], label=str(head_num))
        plt.legend()
        plt.show()

    def plot_circuits(self):
        fig, ax = plt.subplots(self.size, self.s_pairs, sharex=True, sharey=True)
        for circuit_num in range(self.size):
            for s_pair_num in range(self.s_pairs):
                ax[circuit_num, s_pair_num].plot(np.array(self.head_out)[:, circuit_num],
                                                 label=r"$H_" + str(circuit_num) + "$")
                ax[circuit_num, s_pair_num].plot(np.array(self.shouldnt_out)[:, s_pair_num, circuit_num],
                                                 label=r"$N_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")
                ax[circuit_num, s_pair_num].plot(np.array(self.should_out)[:, s_pair_num, circuit_num],
                                                 label=r"$S_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")

                ax[circuit_num, s_pair_num].legend(loc='lower right')
        plt.show()


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
