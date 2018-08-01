# LIKE 2 BUT WITH NOISE AS IN 1

import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self, time_constant=20, error_pairs=2, normalize_weights=(0,), pos_error_to_head=(1.2, 0), neg_error_to_head=(0.6, 0),
                 learning_rate=0.01, dendrite_threshold=2/3, noise_max_amplitude=0.15, noise_rise_rate=0.0000002,
                 noise_fall_rate=0.0002, noise_fall_threshold=0.5, block_threshold=0.02, log_head=False, log_head_out=True,
                 log_neg_error=False, log_neg_error_diff=False, log_neg_error_out=True, log_pos_error_out=True, log_noise_amplitude=False,
                 log_weights=True):
        self.groups = {}
        self.group_sizes = []
        self.names = []
        self.num_circuits = 0
        self.error_pairs = error_pairs
        self.normalize_weights = normalize_weights  # tuple of indices of the e-pairs where weights are normalized
        self.time_constant = time_constant
        self.fast_time_constant = time_constant/10
        self.pos_error_to_head = pos_error_to_head
        self.neg_error_to_head = neg_error_to_head
        self.learning_rate = learning_rate
        self.k = 3  # constant for the activation function of head neurons

        # parameters for the dendritic nonlinearity
        self.dendrite_threshold = dendrite_threshold
        self.dendrite_slope = 1/(1 - self.dendrite_threshold)
        self.dendrite_offset = -self.dendrite_slope*self.dendrite_threshold

        # down counter for blocking learning when neg_error_diff is above block_threshold
        self.block_count = None
        self.block_threshold = block_threshold

        self.head = None
        self.log_head = log_head
        if log_head: self.head_log = None
        self.head_out = None
        self.log_head_out = log_head_out
        if log_head_out: self.head_out_log = None
        self.neg_error = None
        self.log_neg_error = log_neg_error
        if log_neg_error: self.neg_error_log = None
        self.log_neg_error_diff = log_neg_error_diff
        if log_neg_error_diff: self.neg_error_diff_log = None
        self.neg_error_out = None
        self.log_neg_error_out = log_neg_error_out
        if log_neg_error_out: self.neg_error_out_log = None
        self.pos_error_out = None
        self.log_pos_error_out = log_pos_error_out
        if log_pos_error_out: self.pos_error_out_log = None
        self.noise_max_amplitude = noise_max_amplitude
        self.noise_amplitude = None
        self.log_noise_amplitude = log_noise_amplitude
        if log_noise_amplitude: self.noise_amplitude_log = None
        self.noise_period = 4*time_constant
        self.noise_alpha = 0.98
        self.noise_step_num = 0
        self.noise = None
        self.noise_target = None
        self.noise_rise_rate = noise_rise_rate
        self.noise_fall_rate = noise_fall_rate
        self.noise_fall_threshold = noise_fall_threshold
        self.wta_weights = None
        self.wta_sum = None
        self.weights = None
        self.log_weights = log_weights
        if log_weights: self.weights_log = None
        self.weights_mask = None
        self.ones = None

    def add_group(self, name, num_circuits):
        self.group_sizes.append(num_circuits)
        start = self.num_circuits
        self.num_circuits += num_circuits
        end = self.num_circuits
        self.groups[name] = (start, end)

    def build(self):
        # Construct a list of names and a matrix for the winner take all
        self.names = [None] * self.num_circuits
        self.wta_sum = np.zeros(len(self.groups))
        self.wta_weights = np.zeros((self.num_circuits, len(self.groups)))
        for group_num, (group_name, indices) in enumerate(self.groups.items()):
            self.wta_weights[indices[0]:indices[1], group_num] = 1
            for rel_index, abs_index in enumerate(range(indices[0], indices[1])):
                self.names[abs_index] = group_name + '_' + str(rel_index)

        # Initialize weights mask
        self.weights_mask = np.ones((self.error_pairs, self.num_circuits, self.num_circuits))

        self.ones = np.ones((1, 1, self.num_circuits))

    def connect(self, input_group, output_group, error_pair):
        first_input, last_input = self.groups[input_group]
        first_output, last_output = self.groups[output_group]
        self.weights_mask[error_pair, first_output:last_output, first_input:last_input] = 0

    def initialize(self):
        # Initialize activities
        self.head = np.zeros(self.num_circuits)
        if self.log_head: self.head_log = [self.head]
        self.head_out = np.zeros(self.num_circuits)
        if self.log_head_out: self.head_out_log = [self.head_out]
        self.neg_error = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_neg_error: self.neg_error_log = [self.neg_error]
        if self.log_neg_error_diff: self.neg_error_diff_log = [self.neg_error]
        self.block_count = np.zeros((self.error_pairs, self.num_circuits))
        self.neg_error_out = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_neg_error_out: self.neg_error_out_log = [self.neg_error_out]
        self.pos_error_out = np.zeros((self.error_pairs, self.num_circuits))
        if self.log_pos_error_out: self.pos_error_out_log = [self.pos_error_out]

        # Initialize weights
        self.weights = np.ma.masked_array(np.zeros((self.error_pairs, self.num_circuits, self.num_circuits)),
                                          self.weights_mask)
        if self.log_weights: self.weights_log = [self.weights]

        # Initialize noise variables
        self.noise_amplitude = np.ones(self.num_circuits)*self.noise_max_amplitude
        if self.log_noise_amplitude: self.noise_amplitude_log = [self.noise_amplitude]
        self.noise = np.zeros(self.num_circuits)
        self.noise_target = np.zeros(self.num_circuits)

    def dendrite_nonlinearity(self, input_values):
        return np.where(input_values < self.dendrite_threshold, 0, input_values*self.dendrite_slope
                        + self.dendrite_offset)

    def slow_noise(self):
        """Produces low frequency noise"""
        # update noise_amplitude
        self.noise_amplitude = np.clip(self.noise_amplitude + self.noise_rise_rate
                                       - self.noise_fall_rate*(self.head_out > self.noise_fall_threshold),
                                       0, self.noise_max_amplitude)
        if self.log_noise_amplitude:
            self.noise_amplitude_log.append(self.noise_amplitude)

        # every noise_period steps select a new noise_target in the range [-noise_amplitude, noise_amplitude]
        if self.noise_step_num < self.noise_period:
            self.noise_step_num += 1
        else:
            self.noise_step_num = 0
            self.noise_target = np.random.uniform(-self.noise_amplitude, self.noise_amplitude, self.num_circuits)

        # low-pass filter the noise_target
        self.noise = self.noise_alpha*self.noise + (1 - self.noise_alpha)*self.noise_target

        return self.noise
        
    def step(self, external_input):
        self.wta_sum += (-self.wta_sum + np.dot(self.head_out, self.wta_weights)) / self.fast_time_constant
        self.head = (self.head + (-self.head + 2*self.head_out - np.dot(self.pos_error_to_head, self.pos_error_out)
                                  + np.dot(self.neg_error_to_head, self.neg_error_out)
                                  - np.dot(self.wta_sum, self.wta_weights.T) + self.slow_noise()) / self.time_constant)
        if self.log_head: self.head_log.append(self.head)

        self.head_out = np.clip(np.tanh(self.k*self.head), 0, 1)
        if self.log_head_out: self.head_out_log.append(self.head_out)

        neg_error_diff = - self.neg_error
        neg_error_input = self.dendrite_nonlinearity(np.ma.inner(self.head_out, self.weights))
        self.neg_error = (self.neg_error + (-self.neg_error - self.head_out + external_input + neg_error_input)
                          / self.fast_time_constant)
        neg_error_diff += self.neg_error
        if self.log_neg_error: self.neg_error_log.append(self.neg_error)
        if self.log_neg_error_diff: self.neg_error_diff_log.append(neg_error_diff)

        self.block_count = np.where(np.abs(neg_error_diff) > self.block_threshold, 3 * self.time_constant,
                                    np.maximum(self.block_count - 1, 0))

        self.neg_error_out = np.clip(self.neg_error, 0, 1)
        if self.log_neg_error_out: self.neg_error_out_log.append(self.neg_error_out)
        self.pos_error_out = np.clip(-self.neg_error, 0, 1)
        if self.log_pos_error_out: self.pos_error_out_log.append(self.pos_error_out)

        # Update weights
        weight_update = np.ma.inner(self.neg_error[:, :, np.newaxis], self.head_out[:, np.newaxis])
        weight_update[self.normalize_weights, :, :] += self.weights[self.normalize_weights, :, :]*neg_error_input[self.normalize_weights, :, np.newaxis]*(self.ones-self.head_out)
        weight_update = np.where(self.block_count[:, :, np.newaxis], 0, -self.learning_rate*weight_update)
        self.weights = np.clip(self.weights + weight_update, a_min=0, a_max=None)

        if self.log_weights: self.weights_log.append(self.weights)

    def run(self, num_steps, external_input):
        for step_num in range(num_steps):
            self.step(external_input)

    def plot_traces(self):
        print('plotting...')
        x_label = "Simulation steps"
        fig, ax = plt.subplots(self.num_circuits, self.error_pairs, sharex=True, sharey=True)
        fig.suptitle('Neuronal activities')
        for error_pair_num in range(self.error_pairs):
            for circuit_num in range(self.num_circuits):
                if self.log_head:
                    ax[circuit_num, error_pair_num].plot(np.array(self.head_log)[:, circuit_num], 'deepskyblue',
                                                         label=r'$' + self.names[circuit_num] + '.h_i$')
                if self.log_head_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.head_out_log)[:, circuit_num], 'b',
                                                         label=r'$' + self.names[circuit_num] + '.h$')
                if self.log_neg_error:
                    ax[circuit_num, error_pair_num].plot(np.array(self.neg_error_log)[:, error_pair_num, circuit_num],
                                                         'limegreen', label=r'$' + self.names[circuit_num] + '.n_i$')
                if self.log_neg_error_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.neg_error_out_log)[:, error_pair_num, circuit_num],
                                                         'g', label=r'$' + self.names[circuit_num] + '.n$')
                if self.log_pos_error_out:
                    ax[circuit_num, error_pair_num].plot(np.array(self.pos_error_out_log)[:, error_pair_num, circuit_num],
                                                         'r', label=r'$' + self.names[circuit_num] + '.p$')
                ax[circuit_num, error_pair_num].legend()
        for axes in ax[-1]:
            axes.set_xlabel(x_label)

        if self.log_weights:
            fig_w, ax_w = plt.subplots(self.num_circuits, self.error_pairs)
            fig_w.suptitle('Weights')
            for error_pair_num in range(self.error_pairs):
                for circuit_num in range(self.num_circuits):
                    empty = 1
                    for input_num in range(self.num_circuits):
                        if not self.weights_mask[error_pair_num, circuit_num, input_num]:
                            label = r'$w_{' + self.names[input_num] + r'.h\rightarrow ' + self.names[circuit_num] \
                                    + '.p}$'
                            ax_w[circuit_num, error_pair_num].plot(np.array(self.weights_log)
                                                                   [:, error_pair_num, circuit_num, input_num],
                                                                   label=label)
                            empty = 0
                    if empty:
                        ax_w[circuit_num, error_pair_num].axis('off')
                    else:
                        ax_w[circuit_num, error_pair_num].set_ylim([-1.2, 1.2])
                        ax_w[circuit_num, error_pair_num].legend(loc='lower right', ncol=5)
            for axes in ax_w[-1]:
                axes.set_xlabel(x_label)

        if self.log_noise_amplitude:
            fig_n, ax_n = plt.subplots(len(self.group_sizes))
            fig_n.suptitle('Noise amplitude')
            circuit_num = 0
            for group_num, group_size in enumerate(self.group_sizes):
                for _ in range(group_size):
                    ax_n[group_num].plot(np.array(self.noise_amplitude_log)[:, circuit_num], label=r'$' + self.names[circuit_num] + '.a$')
                    circuit_num += 1
                ax_n[group_num].legend()
            ax_n[-1].set_xlabel(x_label)

        if self.log_neg_error_diff:
            fig_d, ax_d = plt.subplots(self.num_circuits, self.error_pairs)
            for error_pair_num in range(self.error_pairs):
                for circuit_num in range(self.num_circuits):
                    ax_d[circuit_num, error_pair_num].plot(np.array(self.neg_error_diff_log)[:, error_pair_num, circuit_num], label=r'$' + self.names[circuit_num] + '.n_diff$')
                    ax_d[circuit_num, error_pair_num].axhline(self.block_threshold, linestyle=':', color='k')
                    ax_d[circuit_num, error_pair_num].axhline(-self.block_threshold, linestyle=':', color='gray')
        plt.show()
