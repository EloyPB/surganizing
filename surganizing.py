import math
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


class CircuitGroup:
    """Group of mismatch detection neural circuits.

    Attributes:
        name (string): Name of the group.
        names (list(string)): Names of each of the circuits in the group.
        num_circuits (int): Number of circuits.
        num_error_pairs (int): Number of error pairs per circuit.
        pos_error_to_head (list(int)): Weights from the positive error unit to the head unit, for each error pair.
        neg_error_to_head (list(int)): Weights from the negative error unit to the head unit, for each error pair.
        weight_normalizing_pairs (list(int)): Index of the error pairs for which incoming weights are normalized to 1.
        time_constant (float): Time constant for the dynamics of the units.
        fast_time_constant (float): Time constant for the dynamics of fast units.
        default_learning_rate (float): Default learning rate.
        learning_rates (list(float)): Current learning rate for each of the error pairs.
    """

    def __init__(self, name, parameters, num_circuits, num_error_pairs, weight_normalizing_pairs,  log_head=False,
                 log_head_out=True, log_neg_error=False, log_neg_error_diff=False, log_neg_error_out=True,
                 log_pos_error_out=True, log_weights=True, log_noise_amplitude=False):

        self.name = name
        self.names = [name + '_' + str(circuit_num) for circuit_num in range(num_circuits)]

        self.num_circuits = num_circuits
        self.num_error_pairs = num_error_pairs
        self.weight_normalizing_pairs = weight_normalizing_pairs

        self.max_neg_error_drive = parameters.max_neg_error_drive
        self.neg_error_to_head = None
        self.max_pos_error_drive = parameters.max_pos_error_drive
        self.pos_error_to_head = None

        self.time_constant = parameters.time_constant
        self.fast_time_constant = parameters.fast_time_constant

        self.default_learning_rate = parameters.learning_rate
        self.learning_rates = [self.default_learning_rate for _ in range(num_error_pairs)]

        # parameters for the dendritic nonlinearity
        self.dendrite_threshold = parameters.dendrite_threshold
        self.dendrite_slope = 1/(1 - self.dendrite_threshold)
        self.dendrite_offset = -self.dendrite_slope*self.dendrite_threshold

        # these variables indicate which variables are logged
        self.log_head = log_head
        self.log_head_out = log_head_out
        self.log_neg_error = log_neg_error
        self.log_neg_error_diff = log_neg_error_diff
        self.log_neg_error_out = log_neg_error_out
        self.log_pos_error_out = log_pos_error_out
        self.log_weights = log_weights
        self.log_noise_amplitude = log_noise_amplitude

        # head neurons
        self.head = np.zeros(num_circuits)
        if log_head:
            self.head_log = [np.zeros(num_circuits)]

        # head neurons output after activation function
        self.head_out = np.zeros(num_circuits)
        if log_head_out:
            self.head_out_log = [np.zeros(num_circuits)]
        self.activation_function_slope = parameters.activation_function_slope

        self.head_external = np.zeros(num_circuits)
        self.head_external_threshold = parameters.head_external_threshold

        # input to negative error neurons
        self.neg_error_input = np.zeros((num_error_pairs, num_circuits))

        # negative error neurons
        self.neg_error = np.zeros((num_error_pairs, num_circuits))
        if log_neg_error:
            self.neg_error_log = [np.zeros((num_error_pairs, num_circuits))]

        # neg_error(t) - neg_error(t-1), will be used to freeze learning during transients
        self.neg_error_diff = np.zeros((num_error_pairs, num_circuits))
        if log_neg_error_diff:
            self.neg_error_diff_log = [np.zeros((num_error_pairs, num_circuits))]

        # neg_error neurons output after clipping between 0 and 1
        self.neg_error_out = np.zeros((num_error_pairs, num_circuits))
        if log_neg_error_out:
            self.neg_error_out_log = [np.zeros((num_error_pairs, num_circuits))]

        # pos_error neurons output
        # pos_error neurons are not calculated explicitly, instead they are assumed to be equal to -neg_error neurons
        self.pos_error_out = np.zeros((num_error_pairs, num_circuits))
        if log_pos_error_out:
            self.pos_error_out_log = [np.zeros((num_error_pairs, num_circuits))]

        # inhibitory neuron that adds up the activity in the group
        self.inhibition = 0

        # down counter for freezing weights when s_diff is above freeze_threshold
        self.freeze_count = np.zeros((num_error_pairs, num_circuits))
        self.freeze_threshold = parameters.freeze_threshold

        self.target_error_pairs = []  # list of error pairs that receive input from other neuron groups
        # list of lists of input neuron groups for each of the error pairs in target_error_pairs
        self.input_groups = [[] for _ in range(num_error_pairs)]
        self.input_names = [None]*num_error_pairs
        self.weights = [None]*num_error_pairs  # list of weight matrices for each error pair
        self.weights_from = [None] * num_error_pairs  # list of neuron from which to take the weights for each pair
        if log_weights:
            self.weights_log = [None]*num_error_pairs

        self.noise = np.zeros(num_circuits)  # current noise
        # in order to produce low frequency noise, a noise target is selected every noise_period steps and low-pass
        # filtered with parameter noise_alpha
        self.noise_target = np.zeros(num_circuits)
        self.noise_step_num = 0
        self.noise_period = parameters.noise_period
        self.noise_smoothing_factor = parameters.noise_smoothing_factor
        # the noise_target is selected in the range [-noise_amplitude, noise_amplitude]
        self.noise_amplitude = parameters.noise_max_amplitude*np.ones(self.num_circuits)
        # noise_amplitude increases constantly with noise_rise_rate saturating at noise_max_amplitude and
        # decays with noise_fall_rate when h_out is above noise_fall_threshold
        self.noise_max_amplitude = parameters.noise_max_amplitude
        if log_noise_amplitude:
            self.noise_amplitude_log = [parameters.noise_max_amplitude*np.ones(self.num_circuits)]
        self.noise_rise_rate = parameters.noise_rise_rate
        self.noise_fall_rate = parameters.noise_fall_rate

    def enable_connections(self, input_groups, target_error_pair):
        """Enable connections from circuit groups in the list 'input_groups' to the error pair number
        'target_error_pair'.

        Args:
            input_groups (list(CircuitsGroup)): List of input circuit groups.
            target_error_pair (int): Index of the target error pair.
        """
        if target_error_pair not in self.target_error_pairs:
            self.target_error_pairs.append(target_error_pair)
        for input_group in input_groups:
            self.input_groups[target_error_pair].append(input_group)

    def initialize_weights(self):
        """Initialize weight matrices. Must be called after enable_connections and before step!
        """
        for error_pair in self.target_error_pairs:
            input_size = 0
            input_names = []
            for input_group in self.input_groups[error_pair]:
                input_size += input_group.num_circuits
                for circuit_name in input_group.names:
                    input_names.append(circuit_name)
            self.weights[error_pair] = np.zeros((input_size, self.num_circuits))
            self.weights_from[error_pair] = self
            if self.log_weights:
                self.weights_log[error_pair] = [np.zeros((input_size, self.num_circuits))]
            self.input_names[error_pair] = input_names

    def slow_noise(self):
        """Produces low frequency noise.
        """
        # update noise_amplitude
        self.noise_amplitude = np.clip(self.noise_amplitude + self.noise_rise_rate
                                       - self.noise_fall_rate * (self.head_out > self.head_external_threshold),
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
        self.noise = self.noise_smoothing_factor * self.noise + (1 - self.noise_smoothing_factor) * self.noise_target
        return self.noise

    def dendrite_nonlinearity(self, input_values):
        return np.where(input_values < self.dendrite_threshold, 0, input_values*self.dendrite_slope
                        + self.dendrite_offset)

    def learning_off(self, error_pairs):
        for error_pair in error_pairs:
            self.learning_rates[error_pair] = 0

    def learning_on(self, learning_rates=()):
        if len(learning_rates) > 0:
            for error_pair, learning_rate in enumerate(learning_rates):
                self.learning_rates[error_pair] = learning_rate
        else:
            self.learning_rates = [self.default_learning_rate for _ in range(self.num_error_pairs)]

    def set_error_pair_drives(self, error_pair_drives):
        self.neg_error_to_head = np.array(error_pair_drives) * self.max_neg_error_drive
        self.pos_error_to_head = np.array(error_pair_drives) * self.max_pos_error_drive

    def step(self, external_input=None):
        """Run one step of the simulation.
        """
        # update the down counters for freezing weights on neg_error neurons whose activity is changing fast
        # in order to block learning during transients
        self.freeze_count = np.where(np.abs(self.neg_error_diff) > self.freeze_threshold, 3*self.time_constant,
                                     np.maximum(self.freeze_count - 1, 0))

        # calculate inputs to neg_error neurons and update weights
        for error_pair in self.target_error_pairs:
            input_values = []
            for input_group in self.input_groups[error_pair]:
                input_values = np.append(input_values, input_group.head_external)

            self.neg_error_input[error_pair] = self.dendrite_nonlinearity(
                np.dot(input_values, self.weights_from[error_pair].weights[error_pair]))

            if self.learning_rates[error_pair]:
                weight_update = np.where(self.freeze_count[error_pair], 0, np.dot(input_values[:, np.newaxis],
                                                                                  - self.neg_error[error_pair][np.newaxis]))
                if error_pair in self.weight_normalizing_pairs:
                    weight_update -= self.weights[error_pair]*(-input_values[:, np.newaxis] + 1)*self.neg_error_input[error_pair]

                self.weights[error_pair] = np.clip(self.weights[error_pair] + self.learning_rates[error_pair] * weight_update, a_min=0, a_max=None)  # clip weights below 0

            if self.log_weights:
                self.weights_log[error_pair].append(self.weights[error_pair])

        # update the activity of inhibitory neuron
        self.inhibition += (-self.inhibition + np.sum(self.head_out)) / self.fast_time_constant

        # update activity of head neuron
        self.head = self.head + (-self.head + 2*self.head_out - self.inhibition + self.slow_noise() +
                                 np.dot(self.neg_error_to_head, self.neg_error_out)
                                 - np.dot(self.pos_error_to_head, self.pos_error_out)) / self.time_constant
        if self.log_head:
            self.head_log.append(self.head)
        self.head_out = np.clip(np.tanh(self.activation_function_slope * self.head), 0, a_max=None)
        if self.log_head_out:
            self.head_out_log.append(self.head_out)

        self.head_external = np.where(self.head > self.head_external_threshold, 1, 0)

        neg_error_update = -self.neg_error - self.head_out + self.neg_error_input
        if external_input is not None:
            neg_error_update += external_input
        self.neg_error_diff = -self.neg_error
        self.neg_error = self.neg_error + neg_error_update / self.fast_time_constant
        self.neg_error_diff += self.neg_error
        if self.log_neg_error_diff:
            self.neg_error_diff_log.append(self.neg_error_diff)
        if self.log_neg_error:
            self.neg_error_log.append(self.neg_error)
        self.neg_error_out = np.clip(self.neg_error, 0, 1)
        if self.log_neg_error_out:
            self.neg_error_out_log.append(self.neg_error_out)
        self.pos_error_out = np.clip(-self.neg_error, 0, 1)
        if self.log_pos_error_out:
            self.pos_error_out_log.append(self.pos_error_out)

    # PLOTTING FUNCTIONS

    def plot_circuits(self, show=False):

        x_label = 'Simulation Steps'

        def get_axis(ax, row, column, num_columns):
            if num_columns == 1 and self.num_circuits == 1:
                return ax
            elif num_columns == 1 or self.num_circuits == 1:
                return ax[max(row, column)]
            else:
                return ax[row, column]

        if self.log_head or self.log_head_out or self.log_neg_error or self.log_neg_error_out or self.log_pos_error_out:
            fig, ax = plt.subplots(self.num_circuits, self.num_error_pairs, sharex='col', sharey='row', num=self.name + ' Act')
            fig.suptitle("Neural Circuits in Module " + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                for error_pair_num in range(self.num_error_pairs):
                    axis = get_axis(ax, circuit_num, error_pair_num, self.num_error_pairs)
                    if circuit_num == 0:
                        axis.set_title("Error Pair " + str(error_pair_num), size='medium')
                    if circuit_num == self.num_circuits-1:
                        axis.set_xlabel(x_label)
                    if error_pair_num == 0:
                        axis.set_ylabel("Circuit " + str(circuit_num))
                        axis.set_ylim([-0.05, 1.05])
                    if self.log_head:
                        axis.plot(np.array(self.head_log)[:, circuit_num], 'b:', label=r'$' + self.names[circuit_num] + '.h^i$')
                    if self.log_head_out:
                        axis.plot(np.array(self.head_out_log)[:, circuit_num], 'b', label=r'$' + self.names[circuit_num] + '.h$')
                    if self.log_neg_error:
                        axis.plot(np.array(self.neg_error_log)[:, error_pair_num, circuit_num], 'g:',
                                              label=r'$' + self.names[circuit_num] + '.n^i_' + str(error_pair_num) + '$')
                    if self.log_neg_error_out:
                        axis.plot(np.array(self.neg_error_out_log)[:, error_pair_num, circuit_num], 'g',
                                                  label=r'$' + self.names[circuit_num] + '.n_' + str(error_pair_num) + '$')
                    if self.log_pos_error_out:
                        axis.plot(np.array(self.pos_error_out_log)[:, error_pair_num, circuit_num], 'r',
                                              label=r'$' + self.names[circuit_num] + '.p_' + str(error_pair_num) + '$')
                    axis.legend(loc='lower left')

        if self.log_noise_amplitude:
            fig, ax = plt.subplots(self.num_circuits, sharex='col', num=self.name + ' Noise')
            fig.suptitle("Noise Amplitude in Module " + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                axis = get_axis(ax, circuit_num, 0, 1)
                axis.plot(np.array(self.noise_amplitude_log)[:, circuit_num])
                axis.set_ylabel("Circuit " + str(circuit_num))
                if circuit_num == self.num_circuits - 1:
                    axis.set_xlabel(x_label)

        if self.log_neg_error_diff:
            fig, ax = plt.subplots(self.num_circuits, self.num_error_pairs, sharex='col', sharey='row', num=self.name + ' Diff')
            fig.suptitle("N(t) - N(t-1) in Module" + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                for error_pair_num in range(self.num_error_pairs):
                    axis = get_axis(ax, circuit_num, error_pair_num, self.num_error_pairs)
                    if circuit_num == 0:
                        axis.set_title("Error Pair " + str(error_pair_num), size='medium')
                    if circuit_num == self.num_circuits-1:
                        axis.set_xlabel(x_label)
                    if error_pair_num == 0:
                        axis.set_ylabel("Circuit " + str(circuit_num))
                    axis.plot(np.array(self.neg_error_diff_log)[:, error_pair_num, circuit_num], 'r')
                    axis.axhline(self.freeze_threshold, linestyle=':', color='gray')
                    axis.axhline(-self.freeze_threshold, linestyle=':', color='gray')

        if self.log_weights:
            fig, ax = plt.subplots(self.num_circuits, len(self.target_error_pairs), sharex='col', sharey='row', num=self.name + ' W')
            fig.suptitle("Incoming Weights into Module " + self.name, size='large')
            for error_pair_num, error_pair in enumerate(self.target_error_pairs):
                num_input_circuits = self.weights[error_pair].shape[0]
                for circuit_num in range(self.num_circuits):
                    axis = get_axis(ax, circuit_num, error_pair_num, len(self.target_error_pairs))
                    if circuit_num == 0:
                        axis.set_title("Error Pair " + str(error_pair), size='medium')
                    if circuit_num == self.num_circuits - 1:
                        axis.set_xlabel(x_label)
                    if error_pair_num == 0:
                        axis.set_ylabel("Circuit " + str(circuit_num))
                    for input_num in range(num_input_circuits):
                        label = r'$W_{' + self.input_names[error_pair][input_num] + r'.h\rightarrow ' + self.names[circuit_num] + '.n_' + str(error_pair) + '}$'
                        axis.plot(np.array(self.weights_log[error_pair])[:, input_num, circuit_num], label=label)
                    axis.legend(loc='lower right')

        if show:
            plt.show()


class Dummy:
    """Used for connecting to filler groups"""
    def __init__(self):
        self.num_circuits = 1
        self.names = ["dummy"]
        self.head_external = np.ones(1)


class ConvolutionalNet:
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.neuron_groups = []
        self.group_names = []
        self.num_groups = 0
        self.strides = []
        self.kernel_sizes = []
        self.offsets = []
        self.filler = []
        self.dummy = Dummy()

    def stack_layer(self, name, group_parameters, num_features, kernel_size, stride, offset=(0, 0), log_head=False,
                    log_head_out=False, log_neg_error=False, log_neg_error_diff=False, log_neg_error_out=False,
                    log_pos_error_out=False, log_weights=False, log_noise_amplitude=False):

        self.num_groups += 1
        self.group_names.append(name)
        self.kernel_sizes.append(kernel_size)
        self.strides.append(stride)
        self.offsets.append(offset)

        if len(self.neuron_groups) == 0:
            input_height = self.image_height
            input_width = self.image_width
        else:
            input_height = len(self.neuron_groups[-1])
            input_width = len(self.neuron_groups[-1][-1])
            self.filler.append(np.ones((len(self.neuron_groups[-1]), len(self.neuron_groups[-1][-1]))))

        # check sizes
        if (input_height % stride[0] != 0 or input_width % stride[1] != 0
                or kernel_size[0] % stride[0] != 0 and kernel_size[0] > stride[0]
                or kernel_size[1] % stride[1] != 0 and kernel_size[1] > stride[1]):
            sys.exit("Input or kernel_size sizes are not multiples of the stride in layer " + name)

        # create new layer
        neuron_groups = []
        for y_out in range(math.ceil((input_height-offset[0])/stride[0])):
            row_of_groups = []
            for x_out in range(math.ceil((input_width-offset[1])/stride[1])):
                group_name = name + "[" + str(y_out) + ", " + str(x_out) + "]"
                new_group = CircuitGroup(group_name, group_parameters, num_features, num_error_pairs=2,
                                         weight_normalizing_pairs=[0], log_head=log_head, log_head_out=log_head_out,
                                         log_neg_error=log_neg_error, log_neg_error_diff=log_neg_error_diff,
                                         log_neg_error_out=log_neg_error_out, log_pos_error_out=log_pos_error_out,
                                         log_weights=log_weights, log_noise_amplitude=log_noise_amplitude)
                new_group.set_error_pair_drives([1, 0])
                # connect previous layer to new layer
                if len(self.neuron_groups) != 0:
                    for y_in in range(y_out*stride[0] + offset[0], y_out*stride[0] + offset[0] + kernel_size[0]):
                        y_in_periodic = y_in if y_in < input_height else y_in - input_height
                        for x_in in range(x_out*stride[1] + offset[1], x_out*stride[1] + offset[1] + kernel_size[1]):
                            x_in_periodic = x_in if x_in < input_width else x_in - input_width
                            new_group.enable_connections([self.neuron_groups[-1][y_in_periodic][x_in_periodic]], 0)
                            self.filler[-1][y_in_periodic][x_in_periodic] = 0
                row_of_groups.append(new_group)
            neuron_groups.append(row_of_groups)

        # connect new layer to previous layer
        if len(self.neuron_groups) != 0:
            for y_in in range(input_height):
                first_y_out = (y_in - offset[0])//stride[0] - max(kernel_size[0]//stride[0] - 1, 0)
                last_y_out = (y_in - offset[0])//stride[0] + 1
                for x_in in range(input_width):
                    if self.filler[-1][y_in][x_in] == 0:
                        first_x_out = (x_in - offset[1])//stride[1] - max(kernel_size[1]//stride[1] - 1, 0)
                        last_x_out = (x_in - offset[1])//stride[1] + 1
                        for y_out in range(first_y_out, last_y_out):
                            for x_out in range(first_x_out, last_x_out):
                                self.neuron_groups[-1][y_in][x_in].enable_connections([neuron_groups[y_out][x_out]], 1)
                    else:
                        self.neuron_groups[-1][y_in][x_in].enable_connections([self.dummy], 1)

        self.neuron_groups.append(neuron_groups)
        print("Created layer " + name + " of size: (" + str(len(neuron_groups)) + ", " + str(len(neuron_groups[0])) + ")")

    def initialize(self):
        for neuron_groups in self.neuron_groups:
            for row_of_groups in neuron_groups:
                for group in row_of_groups:
                    group.initialize_weights()

    def share_weights(self):
        for layer_num, neuron_groups in enumerate(self.neuron_groups):
            for y, row_of_groups in enumerate(neuron_groups):
                for x, group in enumerate(row_of_groups):
                    if y != 0 or x != 0:
                        group.learning_off([0])
                        group.weights_from[0] = neuron_groups[0][0]

                    if layer_num < self.num_groups - 1:
                        kernel_y = self.kernel_sizes[layer_num + 1][0]
                        kernel_x = self.kernel_sizes[layer_num + 1][1]
                        stride_y = self.strides[layer_num + 1][0]
                        stride_x = self.strides[layer_num + 1][1]
                        offset_x = self.offsets[layer_num + 1][0]
                        offset_y = self.offsets[layer_num + 1][1]

                        if self.filler[layer_num][y][x]:
                            y_filler = np.where(self.filler[layer_num])[0][0]
                            x_filler = np.where(self.filler[layer_num])[1][0]
                            if y != y_filler or x != y_filler:
                                group.learning_off([1])
                                group.weights_from[1] = neuron_groups[y_filler][x_filler]
                        elif (y - offset_y) >= kernel_y or (x - offset_x) >= kernel_x:
                            group.learning_off([1])
                            group.weights_from[1] = neuron_groups[(y - offset_y) % stride_y + offset_y][(x - offset_x) % stride_x + offset_x]

    def save_weights(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for layer_num, neuron_groups in enumerate(self.neuron_groups):
            if layer_num > 0:
                neuron_groups[0][0].weights[0].dump(folder_name + "/" + self.group_names[layer_num] + "_(0,0)_0")
            if layer_num < self.num_groups - 1:
                for y in range(self.offsets[layer_num + 1][0], self.offsets[layer_num + 1][0] + self.kernel_sizes[layer_num + 1][0]):
                    for x in range(self.offsets[layer_num + 1][1], self.offsets[layer_num + 1][1] + self.kernel_sizes[layer_num + 1][1]):
                        neuron_groups[y][x].weights[1].dump(folder_name + "/" + self.group_names[layer_num] + "_(" + str(y) + "," + str(x) + ")_1")
                if self.filler[layer_num].any():
                    y_filler = np.where(self.filler[layer_num])[0][0]
                    x_filler = np.where(self.filler[layer_num])[1][0]
                    neuron_groups[y_filler][x_filler].weights[1].dump(folder_name + "/" + self.group_names[layer_num] + "_(" + str(y_filler) + "," + str(x_filler) + ")_1")

    def load_weights(self, folder_name, layers_and_weights="all"):
        if not os.path.exists(folder_name):
            sys.exit("Weights not found! Exiting...")
        if layers_and_weights == "all":
            layers_and_weights = [[group_num, [0, 1]] for group_num in range(self.num_groups)]
        for layer_and_weights in layers_and_weights:
            layer_num = layer_and_weights[0]
            for error_pair in layer_and_weights[1]:
                if error_pair == 0 and layer_num > 0:
                    self.neuron_groups[layer_num][0][0].weights[0] = np.load(folder_name + "/" + self.group_names[layer_num] + "_(0,0)_0", allow_pickle=True)
                elif error_pair == 1 and layer_num < self.num_groups - 1:
                    for y in range(self.offsets[layer_num + 1][0],
                                   self.offsets[layer_num + 1][0] + self.kernel_sizes[layer_num + 1][0]):
                        for x in range(self.offsets[layer_num + 1][1],
                                       self.offsets[layer_num + 1][1] + self.kernel_sizes[layer_num + 1][1]):
                            self.neuron_groups[layer_num][y][x].weights[1] = np.load(
                                folder_name + "/" + self.group_names[layer_num] + "_(" + str(y) + "," + str(x) + ")_1",
                                allow_pickle=True)
                    if self.filler[layer_num].any():
                        y_filler = np.where(self.filler[layer_num])[0][0]
                        x_filler = np.where(self.filler[layer_num])[1][0]
                        for y, x in zip(np.where(self.filler[layer_num])[0], np.where(self.filler[layer_num])[1]):
                            self.neuron_groups[layer_num][y][x].weights[1] = np.load(
                                folder_name + "/" + self.group_names[layer_num] + "_(" + str(y_filler) + "," + str(
                                    x_filler) + ")_1", allow_pickle=True)

    def learning_off(self, layers_and_pairs):
        for layer_and_pairs in layers_and_pairs:
            layer_num = layer_and_pairs[0]
            for row_of_groups in self.neuron_groups[layer_num]:
                for group in row_of_groups:
                    group.learning_off(layer_and_pairs[1])

    def black_and_white(self, input_image):
        external_input = np.zeros((self.image_height, self.image_width, 2, 2))
        external_input[:, :, 0, 0] = input_image
        external_input[:, :, 0, 1] = 1 - input_image
        return external_input

    def run(self, input_image=None, simulation_steps=1, layers=0):
        if input_image is not None:
            external_input = self.black_and_white(input_image)
        else:
            external_input = np.zeros((self.image_height, self.image_width, 2, 2))
        if layers == 0:
            layers = self.num_groups
        for step_num in range(simulation_steps):
            print(step_num)
            for layer in range(layers):
                for row_num, row_of_groups in enumerate(self.neuron_groups[layer]):
                    for col_num, group in enumerate(row_of_groups):
                        if layer == 0:
                            group.step(external_input[row_num, col_num])
                        else:
                            group.step()

    def plot(self, plot_weights=False, show=False):
        # plot activities
        reds = colors.LinearSegmentedColormap.from_list('reds', [(1, 0, 0, 0), (1, 0, 0, 1)], N=100)
        greens = colors.LinearSegmentedColormap.from_list('greens', [(0, 1, 0, 0), (0, 1, 0, 1)], N=100)
        blues = colors.LinearSegmentedColormap.from_list('blues', [(0, 0, 1, 0), (0, 0, 1, 1)], N=100)

        def pretty_weights(group_name, error_pair):
            fig_w, ax_w = plt.subplots()
            plot = ax_w.matshow(weights_reshaped)
            ax_w.set_xticks([i - 0.5 for i in range(weights_reshaped.shape[1])], minor='true')
            ax_w.set_xticks([i for i in range(weights_reshaped.shape[1])])
            ax_w.set_xticklabels([i for i in range(kernel_x)] * output_features)
            ax_w.xaxis.set_ticks_position('bottom')
            ax_w.set_yticks([i - 0.5 for i in range(weights_reshaped.shape[0])], minor='true')
            ax_w.set_yticks([i for i in range(weights_reshaped.shape[0])])
            ax_w.set_yticklabels([i for i in range(kernel_y)] * input_features)
            ax_w.grid(which='minor', linestyle='solid', color='gray')
            ax_w.set_title("Weights onto the " + error_pair + " error pair of '" + group_name + "'")
            fig_w.colorbar(plot, ax=ax_w)

            for boundary_num in range(output_features - 1):
                ax_w.axvline(kernel_x + kernel_x * boundary_num - 0.5, color='w')
            for boundary_num in range(input_features - 1):
                ax_w.axhline(kernel_y + kernel_y * boundary_num - 0.5, color='w')
        fig, ax = plt.subplots(1, self.num_groups)

        for layer_num, neuron_groups in enumerate(self.neuron_groups):
            height = len(neuron_groups)
            width = len(neuron_groups[0])
            num_features = neuron_groups[0][0].num_circuits

            head_out = np.zeros((height, width*num_features))
            neg_error_out = np.zeros((height, width*num_features))
            pos_error_out = np.zeros((height, width*num_features))

            for feature_num in range(num_features):
                for row_num, row_of_groups in enumerate(neuron_groups):
                    for col_num, neuron_group in enumerate(row_of_groups):
                        head_out[row_num, col_num+feature_num*width] = neuron_group.head_out[feature_num]
                        neg_error_out[row_num, col_num+feature_num*width] = neuron_group.neg_error_out[-1, feature_num]
                        pos_error_out[row_num, col_num+feature_num*width] = neuron_group.pos_error_out[-1, feature_num]

            ax[layer_num].matshow(head_out, cmap=blues, vmin=0, vmax=1)
            ax[layer_num].matshow(neg_error_out, cmap=greens, vmin=0, vmax=1)
            ax[layer_num].matshow(pos_error_out, cmap=reds, vmin=0, vmax=1)

            ax[layer_num].set_xticks([i - 0.5 for i in range(width*num_features)], minor='true')
            ax[layer_num].set_yticks([i - 0.5 for i in range(height)], minor='true')
            ax[layer_num].grid(which='minor', linestyle='solid', color='gray')

            for boundary_num in range(num_features - 1):
                ax[layer_num].axvline(width + width*boundary_num - 0.5, color='k')

            # plot weights onto first error pair
            if plot_weights and layer_num > 0:
                kernel_y = self.kernel_sizes[layer_num][0]
                kernel_x = self.kernel_sizes[layer_num][1]
                input_features = self.neuron_groups[layer_num - 1][0][0].num_circuits
                output_features = self.neuron_groups[layer_num][0][0].num_circuits

                weights_reshaped = np.zeros((input_features*kernel_y, output_features*kernel_x))
                for input_feature in range(input_features):
                    for output_feature in range(output_features):
                        y_indices = tuple((i for i in range(input_feature, kernel_y*kernel_x*input_features, input_features)))
                        indices = (y_indices, output_feature)
                        weights_reshaped[input_feature*kernel_y:(input_feature+1)*kernel_y, output_feature*kernel_x:(output_feature+1)*kernel_x] = neuron_groups[0][0].weights[0][indices].reshape((kernel_y, kernel_x))

                pretty_weights(self.group_names[layer_num], 'first')

            # plot weights onto second error pair
            if plot_weights and layer_num < self.num_groups - 1:
                kernel_y = self.kernel_sizes[layer_num + 1][0]
                kernel_x = self.kernel_sizes[layer_num + 1][1]
                offset_y = self.offsets[layer_num + 1][0]
                offset_x = self.offsets[layer_num + 1][1]
                input_features = self.neuron_groups[layer_num + 1][0][0].num_circuits
                output_features = self.neuron_groups[layer_num][0][0].num_circuits

                weights_reshaped = np.zeros((input_features*kernel_y, output_features*kernel_x))

                for input_feature in range(input_features):
                    for output_feature in range(output_features):
                        for y in range(kernel_y):
                            for x in range(kernel_x):
                                weights_reshaped[input_feature*kernel_y + y, output_feature*kernel_x + x] = self.neuron_groups[layer_num][y + offset_y][x + offset_x].weights[1][input_feature, output_feature]

                pretty_weights(self.group_names[layer_num], 'second')

        # plot colorbars in the activations figure
        fig.subplots_adjust(right=0.82)
        gradient = np.linspace(0, 1, 100).reshape(100, 1)

        ax_position = ax[0].get_position()
        y_span = ax_position.height

        def pretty_color_bar(axes, color_map, label):
            axes.imshow(gradient, cmap=color_map, aspect='auto', origin='lower')
            axes.axes.set_ylabel(label)
            axes.yaxis.set_label_position("right")
            axes.axes.get_xaxis().set_visible(False)
            axes.yaxis.tick_right()
            axes.yaxis.set_ticks([-0.5, 49.5, 99.5])
            axes.yaxis.set_ticklabels(np.linspace(0, 1, 3))

        p_cbar_ax = fig.add_axes([0.87, ax_position.y0, 0.015, y_span * 4 / 15])
        pretty_color_bar(p_cbar_ax, reds, "positive error")

        pos = ax_position.y0 + y_span * 4 / 15 + y_span / 10
        n_cbar_ax = fig.add_axes([0.87, pos, 0.015, y_span * 4 / 15])
        pretty_color_bar(n_cbar_ax, greens, "negative error")

        pos += y_span * 4 / 15 + y_span / 10
        h_cbar_ax = fig.add_axes([0.87, pos, 0.015, y_span * 4 / 15])
        pretty_color_bar(h_cbar_ax, blues, "head")

        if show:
            plt.show()