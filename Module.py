import numpy as np


class Module:
    def __init__(self, size, tau):
        self.size = size
        self.tau_head = tau
        self.tau_fast = tau / 20

        self.head = np.zeros(size)
        self.head_out = np.zeros(size)
        self.positive = np.zeros((2, size))
        self.positive_out = np.zeros((2, size))
        self.negative_out = np.zeros((2, size))

        self.novelty = 0
        self.inhibition = 0

        self.child = None
        self.parent = None

        self.saturation = np.vectorize(self.saturation, otypes=[np.float])
        self.activation = np.vectorize(self.activation, otypes=[np.float])

    def is_child_of(self, parent):
        assert isinstance(parent, Module)
        self.parent = parent
        parent.child = self

    def step(self, predictions):
        self.head += (-self.head + self.head_out + self.negative_out[0] - self.positive_out[0]) / self.tau_head
        self.head_out = self.activation(self.head)
        self.positive += (-self.positive + self.head_out - predictions) / self.tau_fast
        self.positive_out = self.saturation(self.positive)
        self.negative_out = self.saturation(-self.positive)
        self.novelty += (-self.novelty + np.sum(self.positive_out[1])) / self.tau_fast
        self.inhibition += (-self.inhibition + np.sum(self.head_out)) / self.tau_fast

    def saturation(self, value):
        """Saturates 'value' between 0 and 1"""
        if value < 0:
            return 0
        elif value > 1:
            return 1
        else:
            return value

    def activation(self, value):
        """Activation function for the head neurons"""
        if value <= 0.25:
            return 0
        elif value <= 0.75:
            return 2 * value - 0.5
        else:
            return 1
