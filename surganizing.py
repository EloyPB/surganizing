import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import matplotlib.colors as colors


class NeuronGroup:
    def __init__(self, name, size, num_s_pairs=2, s_pair_weights=[1, 0], time_constant=20, noise_max_amplitude=0.15,
                 dendrite_threshold=0.6, log_h=False, log_h_out=False, log_s=False, log_s_diff=False, log_s_out=False,
                 log_sn_out=False, log_inhibition=False, log_weights=True, log_noise_amplitude=False):

        self.name = name
        self.size = size
        self.num_s_pairs = num_s_pairs
        self.s_pair_weights = np.array(s_pair_weights)
        self.time_constant = time_constant
        self.fast_time_constant = time_constant / 10
        self.dendrite_threshold = dendrite_threshold
        self.dendrite_slope = 1/(1 - self.dendrite_threshold)
        self.dendrite_offset = -self.dendrite_slope*self.dendrite_threshold
        self.log_h = log_h
        self.log_h_out = log_h_out
        self.log_s = log_s
        self.log_s_diff = log_s_diff
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

        self.block_count = np.zeros((num_s_pairs, size))
        self.block_threshold = 0.02  # this has to be adjusted by hand

        self.s_input = np.zeros((num_s_pairs, size))

        self.s = np.zeros((num_s_pairs, size))
        if log_s:
            self.s_log = [np.zeros((num_s_pairs, size))]

        self.s_diff = np.zeros((num_s_pairs, size))
        if log_s_diff:
            self.s_diff_log = [np.zeros((num_s_pairs, size))]

        self.s_out = np.zeros((num_s_pairs, size))
        if log_s_out:
            self.s_out_log = [np.zeros((num_s_pairs, size))]

        self.sn_out = np.zeros((num_s_pairs, size))
        if log_sn_out:
            self.sn_out_log = [np.zeros((num_s_pairs, size))]

        self.inhibition = 0
        if log_inhibition:
            self.inhibition_log = [0]

        self.to_s_pairs = []
        self.from_modules = [[] for _ in range(num_s_pairs)]
        self.weights = [None]*num_s_pairs
        self.weight_decay_rate = 0
        if log_weights:
            self.weights_log = [None]*num_s_pairs

        self.noise = 0
        self.noise_target = 0
        self.noise_step = 0
        self.noise_period = 4*time_constant
        self.noise_alpha = 0.98
        self.noise_amplitude = noise_max_amplitude*np.ones(self.size)
        self.noise_max_amplitude = noise_max_amplitude
        self.noise_rise_rate = 0.0000002
        self.noise_fall_rate = 0.0002
        if log_noise_amplitude:
            self.noise_amplitude_log = [noise_max_amplitude*np.ones(self.size)]

    def slow_noise(self):
        self.noise_amplitude = np.clip(self.noise_amplitude + self.noise_rise_rate
                                       - self.noise_fall_rate*(self.h_out>0.5), 0, self.noise_max_amplitude)
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
        for from_module in from_modules:
            self.from_modules[to_s_pair].append(from_module)

    def initialize_weights(self):
        for s_pair in self.to_s_pairs:
            input_size = 0
            for module in self.from_modules[s_pair]:
                input_size += module.size
            self.weights[s_pair] = np.zeros((input_size, self.size))
            if self.log_weights:
                self.weights_log[s_pair] = [np.zeros((input_size, self.size))]

    def step(self, s_input, learning_rate):
        # update blocked states
        self.block_count = np.where(np.abs(self.s_diff) > self.block_threshold, 3*self.time_constant,
                                    np.maximum(self.block_count - 1, 0))

        # calculate input to 's' neurons and update weights
        for s_pair in self.to_s_pairs:
            input_values = []
            for module in self.from_modules[s_pair]:
                input_values = np.append(input_values, module.h_out)

            self.s_input[s_pair] = self.dendritic_nonlinearity(np.dot(input_values, self.weights[s_pair]))

            self.weights[s_pair] += np.where(self.block_count[s_pair], 0,
                                             learning_rate*np.dot(input_values[np.newaxis].transpose(),
                                                                        -self.s[s_pair][np.newaxis]))
            if self.s_pair_weights[s_pair] > 0:
                self.weights[s_pair] -= np.where(np.vstack([self.s_input[s_pair] for _ in range(len(input_values))])
                                                 > 1.1*np.dstack([input_values for _ in range(self.size)])[0],
                                                 learning_rate, 0)
            self.weights[s_pair] -= self.weights[s_pair]*self.weight_decay_rate*learning_rate
            self.weights[s_pair] = np.where(self.weights[s_pair] < 0, 0, self.weights[s_pair])

            if self.log_weights:
                self.weights_log[s_pair].append(deepcopy(self.weights[s_pair]))

            self.s_input[s_pair] = self.dendritic_nonlinearity(np.dot(input_values, self.weights[s_pair]))

        # update the activity of neurons
        self.inhibition += (-self.inhibition + np.sum(self.h_out)) / self.fast_time_constant
        if self.log_inhibition:
            self.inhibition_log.append(self.log_inhibition)

        self.h += (-self.h + 2*self.h_out - self.inhibition + np.dot(0.5*self.s_pair_weights, self.s_out)  # make this parameter explicit
                   - np.dot(self.s_pair_weights, self.sn_out) + self.slow_noise()) / self.time_constant
        if self.log_h:
            self.h_log.append(self.h)
        self.h_out = np.clip(np.tanh(3*self.h), 0, 1)  # make this parameter explicit
        if self.log_h_out:
            self.h_out_log.append(self.h_out)

        s_update = -self.s - self.h_out + self.s_input
        if s_input is not None:
            s_update += s_input
        self.s_diff = -self.s
        self.s += s_update / self.fast_time_constant
        self.s_diff += self.s
        if self.log_s_diff:
            self.s_diff_log.append(self.s_diff)
        if self.log_s:
            self.s_log.append(self.s)
        self.s_out = np.clip(self.s, 0, 1)
        if self.log_s_out:
            self.s_out_log.append(self.s_out)
        self.sn_out = np.clip(-self.s, 0, 1)
        if self.log_sn_out:
            self.sn_out_log.append(self.sn_out)

    def dendritic_nonlinearity(self, input_values):
        return np.where(input_values < self.dendrite_threshold, 0, input_values*self.dendrite_slope
                        + self.dendrite_offset)

    # PLOTTING FUNCTIONS

    def plot_heads(self):
        fig, ax = plt.subplots()
        for h_num in range(self.size):
            ax.plot(np.array(self.h_out_log)[:, h_num], label=str(h_num))
        plt.legend()
        plt.show()

    def plot_circuits(self, show):
        if self.log_h_out or self.log_s_out or self.log_sn_out:
            fig, ax = plt.subplots(self.size, self.num_s_pairs, sharex=True, sharey=True)
            fig.suptitle("Neural Circuits in Module " + self.name, size='large')
            for circuit_num in range(self.size):
                for s_pair_num in range(self.num_s_pairs):
                    axes_indices = self.axes_indices(circuit_num, s_pair_num, self.num_s_pairs)
                    if circuit_num == 0:
                        ax[axes_indices].set_title("S-Pair " + str(s_pair_num), size='medium')
                    if s_pair_num == 0:
                        ax[axes_indices].set_ylabel("Circuit " + str(circuit_num))
                        ax[axes_indices].set_ylim([-0.2, 1.2])
                    if self.log_h_out:
                        ax[axes_indices].plot(np.array(self.h_out_log)[:, circuit_num], 'b',
                                              label=r"$H_" + str(circuit_num) + "$")
                    if self.log_sn_out:
                        ax[axes_indices].plot(np.array(self.sn_out_log)[:, s_pair_num, circuit_num], 'r',
                                              label=r"$N_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")
                    if self.log_s_out:
                        ax[axes_indices].plot(np.array(self.s_out_log)[:, s_pair_num, circuit_num], 'g',
                                              label=r"$S_{" + str(circuit_num) + "," + str(s_pair_num) + "}$")
                    ax[axes_indices].legend(loc='lower left')

        if self.log_noise_amplitude:
            fig, ax = plt.subplots(self.size, sharex=True)
            fig.suptitle("Noise Amplitude in Module " + self.name, size='large')
            for circuit_num in range(self.size):
                ax[circuit_num].plot(np.array(self.noise_amplitude_log)[:, circuit_num])
                ax[circuit_num].set_ylabel("Circuit " + str(circuit_num))

        if self.log_s_diff:
            fig, ax = plt.subplots(self.size, self.num_s_pairs, sharex=True, sharey=True)
            fig.suptitle("S(t) - S(t-1) in Module" + self.name, size='large')
            for circuit_num in range(self.size):
                for s_pair_num in range(self.num_s_pairs):
                    axes_indices = self.axes_indices(circuit_num, s_pair_num, self.num_s_pairs)
                    if circuit_num == 0:
                        ax[axes_indices].set_title("S-Pair " + str(s_pair_num), size='medium')
                    if s_pair_num == 0:
                        ax[axes_indices].set_ylabel("Circuit " + str(circuit_num))
                    ax[axes_indices].plot(np.array(self.s_diff_log)[:, s_pair_num, circuit_num], 'r',
                                          label=r"$S_{diff," + str(circuit_num) + "," + str(s_pair_num) + "}$")
                    ax[axes_indices].legend(loc='lower right')

        if self.log_weights:
            fig, ax = plt.subplots(self.size, len(self.to_s_pairs), sharex=True, sharey=True)
            fig.suptitle("Incoming Weights in Module " + self.name, size='large')
            for s_pair_num, s_pair in enumerate(self.to_s_pairs):
                input_size = self.weights[s_pair].shape[0]
                for circuit_num in range(self.size):
                    axes_indices = self.axes_indices(circuit_num, s_pair_num, len(self.to_s_pairs))
                    if circuit_num == 0:
                        ax[axes_indices].set_title("S-Pair " + str(s_pair), size='medium')
                    if s_pair_num == 0:
                        ax[axes_indices].set_ylabel("Circuit " + str(circuit_num))
                    for input_num in range(input_size):
                        ax[axes_indices].plot(np.array(self.weights_log[s_pair])[:, input_num, circuit_num],
                                              label=r"$W_{H_" + str(input_num) +r"\rightarrow S_" + str(circuit_num) + "}$")
                    ax[axes_indices].legend(loc='lower right')

        if show:
            plt.show()

    def axes_indices(self, circuit_num, s_pair_num, max_s_pairs):
        if max_s_pairs == 1 or self.size == 1:
            return max(circuit_num, s_pair_num)
        else:
            return circuit_num, s_pair_num


class FeatureMaps:
    def __init__(self, name):
        self.name = name
        self.neuron_groups = []
        self.height = 0
        self.width = 0

    def build_initial(self, height, width, num_features, num_s_pairs=2, s_pair_weights=[1, 0], time_constant=20,
                      noise_max_amplitude=0.15, dendrite_threshold=0.6, log_h=False, log_h_out=True, log_s=False,
                      log_s_diff=False, log_s_out=True, log_sn_out=True, log_inhibition=False, log_weights=True,
                      log_noise_amplitude=False):

        self.height = height
        self.width = width
        self.neuron_groups = [[[] for _ in range(width)] for _ in range(height)]
        for row_num in range(height):
            for col_num in range(width):
                group_name = self.name + " " + str(row_num) + "," + str(col_num)
                self.neuron_groups[row_num][col_num] = NeuronGroup(group_name, num_features, num_s_pairs, s_pair_weights,
                                                                   time_constant, noise_max_amplitude,
                                                                   dendrite_threshold, log_h, log_h_out,
                                                                   log_s, log_s_diff, log_s_out, log_sn_out,
                                                                   log_inhibition, log_weights, log_noise_amplitude)

    def build_on_top(self, input_feature_maps, kernel_height, kernel_width, stride, num_features, num_s_pairs=2,
                     s_pair_weights=[1, 0], time_constant=20, noise_max_amplitude=0.15, dendrite_threshold=0.6,
                     log_h=False, log_h_out=True, log_s=False, log_s_diff=False, log_s_out=True, log_sn_out=True,
                     log_inhibition=False, log_weights=True, log_noise_amplitude=False):

        self.neuron_groups = []
        y = 0
        row_num_out = 0
        while y <= input_feature_maps.height - kernel_height:
            x = 0
            col_num_out = 0
            new_row = []
            while x <= input_feature_maps.width - kernel_width:
                group_name = self.name + " " + str(row_num_out) + "," + str(col_num_out)
                new_group = NeuronGroup(group_name, num_features, num_s_pairs, s_pair_weights, time_constant,
                                        noise_max_amplitude, dendrite_threshold, log_h, log_h_out, log_s, log_s_diff,
                                        log_s_out, log_sn_out, log_inhibition, log_weights, log_noise_amplitude)
                new_row.append(new_group)
                for row_num_in in range(y, y + kernel_height):
                    for col_num_in in range(x, x + kernel_width):
                        new_group.enable_connections(0, [input_feature_maps.neuron_groups[row_num_in][col_num_in]])
                        input_feature_maps.neuron_groups[row_num_in][col_num_in].enable_connections(1, [new_group])
                x += stride
                col_num_out += 1
            self.neuron_groups.append(new_row)
            y += stride
            row_num_out += 1
        self.height = len(self.neuron_groups)
        self.width = len(self.neuron_groups[0])


class ConvNet:
    def __init__(self, all_feature_maps):
        self.all_feature_maps = all_feature_maps
        self.max_num_maps = 0
        for feature_maps in all_feature_maps:
            for row_of_groups in feature_maps.neuron_groups:
                for neuron_group in row_of_groups:
                    neuron_group.initialize_weights()
                    if neuron_group.size > self.max_num_maps:
                        self.max_num_maps = neuron_group.size

    def run_step(self, input_pattern, learning_rates):
        for layer_num, feature_maps in enumerate(self.all_feature_maps):
            for row_num, row_of_groups in enumerate(feature_maps.neuron_groups):
                for col_num, neuron_group in enumerate(row_of_groups):
                    if layer_num == 0:
                        s_input = np.zeros((neuron_group.num_s_pairs, neuron_group.size))
                        s_input[0] = input_pattern[row_num, col_num]
                    else:
                        s_input = None
                    neuron_group.step(s_input, learning_rates[layer_num])

    def plot_activities(self):
        reds = colors.LinearSegmentedColormap.from_list('reds', [(1, 0, 0, 0), (1, 0, 0, 1)], N=100)
        greens = colors.LinearSegmentedColormap.from_list('greens', [(0, 1, 0, 0), (0, 1, 0, 1)], N=100)
        blues = colors.LinearSegmentedColormap.from_list('blues', [(0, 0, 1, 0), (0, 0, 1, 1)], N=100)

        fig, ax = plt.subplots(self.max_num_maps, len(self.all_feature_maps))

        for layer_num, feature_maps in enumerate(self.all_feature_maps):
            height = len(feature_maps.neuron_groups)
            width = len(feature_maps.neuron_groups[0])
            num_maps = feature_maps.neuron_groups[0][0].size
            h_out = np.zeros((height, width, num_maps))
            s_out = np.zeros((height, width, num_maps))
            sn_out = np.zeros((height, width, num_maps))
            for row_num, row_of_groups in enumerate(feature_maps.neuron_groups):
                for col_num, neuron_group in enumerate(row_of_groups):
                    h_out[row_num, col_num] = neuron_group.h_out
                    s_out[row_num, col_num] = neuron_group.s_out[-1]
                    sn_out[row_num, col_num] = neuron_group.sn_out[-1]

            for map_num in range(num_maps):
                ax[map_num, layer_num].matshow(h_out[:, :, map_num], cmap=blues, vmin=0, vmax=1)
                ax[map_num, layer_num].matshow(s_out[:, :, map_num], cmap=greens, vmin=0, vmax=1)
                ax[map_num, layer_num].matshow(sn_out[:, :, map_num], cmap=reds, vmin=0, vmax=1)

                ax[map_num, layer_num].set_xticks([x - 0.5 for x in ax[map_num, layer_num].get_xticks()][1:],
                                                  minor='true')
                ax[map_num, layer_num].set_yticks([y - 0.5 for y in ax[map_num, layer_num].get_yticks()][1:],
                                                  minor='true')
                ax[map_num, layer_num].grid(which='minor', linestyle='solid')

        plt.show()

