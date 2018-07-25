import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Network:
    def __init__(self, time_constant=20, error_pairs=2, pos_error_to_head=(2, 0), neg_error_to_head=(0.3, 0),
                 dendrite_threshold=2/3, log_head=False, log_head_out=True, log_pos_error=False, log_pos_error_out=True,
                 log_neg_error_out=True):
        self.groups = {}
        self.names = []
        self.num_circuits = 0
        self.error_pairs = error_pairs
        self.time_constant = time_constant
        self.fast_time_constant = time_constant/10
        self.pos_error_to_head = pos_error_to_head
        self.neg_error_to_head = neg_error_to_head
        self.k = 3  # constant for the activation function of head neurons

        # parameters for the dendritic nonlinearity
        self.dendrite_threshold = dendrite_threshold
        self.dendrite_slope = 1/(1 - self.dendrite_threshold)
        self.dendrite_offset = -self.dendrite_slope*self.dendrite_threshold

        self.head = None
        self.log_head = log_head
        if log_head: self.head_log = None
        self.head_out = None
        self.log_head_out = log_head_out
        if log_head_out: self.head_out_log = None
        self.pos_error = None
        self.log_pos_error = log_pos_error
        if log_pos_error: self.pos_error_log = None
        self.pos_error_out = None
        self.log_pos_error_out = log_pos_error_out
        if log_pos_error_out: self.pos_error_out_log = None
        self.neg_error_out = None
        self.log_neg_error_out = log_neg_error_out
        if log_neg_error_out: self.neg_error_out_log = None

    def add_group(self, name, num_circuits):
        start = self.num_circuits
        self.num_circuits += num_circuits
        end = self.num_circuits
        self.groups[name] = (start, end)

    def initialize_activities(self):
        self.names = [None]*self.num_circuits
        for group_name, indices in self.groups.items():
            for rel_index, abs_index in enumerate(range(indices[0], indices[1])):
                self.names[abs_index] = r'$' + group_name + '_' + str(rel_index)
        self.head = np.zeros(self.num_circuits)
        if self.log_head: self.head_log = [self.head]
        self.head_out = np.zeros(self.num_circuits)
        if self.log_head_out: self.head_out_log = [self.head_out]
        self.pos_error = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_pos_error: self.pos_error_log = [self.pos_error]
        self.pos_error_out = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_pos_error_out: self.pos_error_out_log = [self.pos_error_out]
        self.neg_error_out = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_neg_error_out: self.neg_error_out_log = [self.neg_error_out]
        print(self.names)

    def dendrite_nonlinearity(self, input_values):
        return np.where(input_values < self.dendrite_threshold, 0, input_values*self.dendrite_slope
                        + self.dendrite_offset)
        
    def step(self, external_input):
        self.head += (-self.head + self.head_out - np.dot(self.pos_error_to_head, self.pos_error_out)
                      + np.dot(self.neg_error_to_head, self.neg_error_out)) / self.time_constant
        if self.log_head: self.head_log.append(self.head)

        self.head_out = np.clip(np.tanh(self.k*self.head), 0, 1)
        if self.log_head_out: self.head_out_log.append(self.head_out)

        self.pos_error += (-self.pos_error + self.head_out - external_input) / self.fast_time_constant
        if self.log_pos_error: self.pos_error_log.append(self.pos_error)

        self.pos_error_out = np.clip(self.pos_error, 0, 1)
        if self.log_pos_error_out: self.pos_error_out_log.append(self.pos_error_out)
        self.neg_error_out = np.clip(-self.pos_error, 0, 1)
        if self.log_neg_error_out: self.neg_error_out_log.append(self.neg_error_out)

    def run(self, num_steps, external_input):
        for step_num in range(num_steps):
            self.step(external_input)

    def plot_traces(self):
        fig, ax = plt.subplots(self.num_circuits, self.error_pairs, sharex=True, sharey=True)
        for error_pair_num in range(self.error_pairs):
            for circuit_num in range(self.num_circuits):
                if self.log_head:
                    ax[circuit_num, error_pair_num].plot(np.array(self.head_log)[:, circuit_num], 'deepskyblue',
                                                         label=self.names[circuit_num] + ' h_i$')
                if self.log_head_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.head_out_log)[:, circuit_num], 'b',
                                                         label=self.names[circuit_num] + ' h$')
                if self.log_pos_error:
                    ax[circuit_num, error_pair_num].plot(np.array(self.pos_error_log)[:, error_pair_num, circuit_num],
                                                         'orange', label=self.names[circuit_num] + ' p_i$')
                if self.log_pos_error_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.pos_error_out_log)[:, error_pair_num, circuit_num],
                                                         'r', label=self.names[circuit_num] + ' p$')
                if self.log_neg_error_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.neg_error_out_log)[:, error_pair_num, circuit_num],
                                                         'g', label=self.names[circuit_num] + ' n$')
                ax[circuit_num, error_pair_num].legend()
        plt.show()