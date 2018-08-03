import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


class NeuronGroup:
    """Initialization function"""
    def __init__(self, name, num_circuits, num_error_pairs=2, pos_error_to_head=(1, 0), neg_error_to_head=(0.5, 0),
                 normalize_weights=(0,), time_constant=20, learning_rate=0.01, noise_max_amplitude=0.15, noise_rise_rate=0.0000002,
                 noise_fall_rate=0.0002, noise_fall_threshold=0.5,  dendrite_threshold=2/3, freeze_threshold=0.02, 
                 log_head=False, log_head_out=True, log_neg_error=False, log_neg_error_diff=False, log_neg_error_out=True,
                 log_pos_error_out=True, log_weights=True, log_noise_amplitude=False):

        self.name = name  # name of the neuron group
        self.names = [name[0] + '_' + str(circuit_num) for circuit_num in range(num_circuits)]
        self.num_circuits = num_circuits  # number of mismatch detection circuits in the neuron group
        self.num_error_pairs = num_error_pairs
        self.pos_error_to_head = pos_error_to_head
        self.neg_error_to_head = neg_error_to_head
        self.normalize_weights = normalize_weights
        self.time_constant = time_constant
        self.fast_time_constant = time_constant/10
        self.default_learning_rate = learning_rate
        self.learning_rate = learning_rate

        # parameters for the dendritic nonlinearity
        self.dendrite_threshold = dendrite_threshold
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
        self.k = 3  # constant for the activation function

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

        # down counter for freezeing weights when s_diff is above freeze_threshold
        self.freeze_count = np.zeros((num_error_pairs, num_circuits))
        self.freeze_threshold = freeze_threshold

        self.target_error_pairs = []  # list of error pairs that receive input from other neuron groups
        # list of lists of input neuron groups for each of the error pairs in target_error_pairs
        self.input_groups = [[] for _ in range(num_error_pairs)]
        self.input_names = [None]*num_error_pairs
        self.weights = [None]*num_error_pairs  # list of weight matrices for each error pair
        self.ones = None
        if log_weights:
            self.weights_log = [None]*num_error_pairs

        self.noise = np.zeros(num_circuits)  # current noise
        # in order to produce low frequency noise, a noise target is selected every noise_period steps and low-pass
        # filtered with parameter noise_alpha
        self.noise_target = np.zeros(num_circuits)
        self.noise_step_num = 0
        self.noise_period = 4*time_constant
        self.noise_alpha = 0.98
        # the noise_target is selected in the range [-noise_amplitude, noise_amplitude]
        self.noise_amplitude = noise_max_amplitude*np.ones(self.num_circuits)
        # noise_amplitude increases constantly with noise_rise_rate saturating at noise_max_amplitude and
        # decays with noise_fall_rate when h_out is above noise_fall_threshold
        self.noise_max_amplitude = noise_max_amplitude
        if log_noise_amplitude:
            self.noise_amplitude_log = [noise_max_amplitude*np.ones(self.num_circuits)]
        self.noise_rise_rate = noise_rise_rate
        self.noise_fall_rate = noise_fall_rate
        self.noise_fall_threshold = noise_fall_threshold

    def enable_connections(self, input_groups, target_error_pair):
        """Enable connections from neuron groups in the list 'input_groups'
        to the error pair number 'target_error_pair'"""
        if target_error_pair not in self.target_error_pairs:
            self.target_error_pairs.append(target_error_pair)
        for input_group in input_groups:
            self.input_groups[target_error_pair].append(input_group)

    def initialize_weights(self):
        """Initialize weight matrices. Must be called after enable_connections and before step!"""
        for error_pair in self.target_error_pairs:
            input_size = 0
            input_names = []
            for input_group in self.input_groups[error_pair]:
                input_size += input_group.num_circuits
                for circuit_name in input_group.names:
                    input_names.append(circuit_name)
            self.weights[error_pair] = np.zeros((input_size, self.num_circuits))
            if self.log_weights:
                self.weights_log[error_pair] = [np.zeros((input_size, self.num_circuits))]
            self.input_names[error_pair] = input_names
            self.ones = np.ones((input_size, 1))

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

    def dendrite_nonlinearity(self, input_values):
        return np.where(input_values < self.dendrite_threshold, 0, input_values*self.dendrite_slope
                        + self.dendrite_offset)

    def learning_off(self):
        self.learning_rate = 0

    def learning_on(self, learning_rate=False):
        self.learning_rate = learning_rate if learning_rate else self.default_learning_rate

    def step(self, external_input=None):
        """Run one step of the simulation"""
        # update the down counters for freezing weights on neg_error neurons whose activity is changing fast
        # in order to block learning during transients
        self.freeze_count = np.where(np.abs(self.neg_error_diff) > self.freeze_threshold, 3*self.time_constant,
                                     np.maximum(self.freeze_count - 1, 0))

        # calculate inputs to neg_error neurons and update weights
        for error_pair in self.target_error_pairs:
            input_values = []
            for module in self.input_groups[error_pair]:
                input_values = np.append(input_values, module.head_out)

            self.neg_error_input[error_pair] = self.dendrite_nonlinearity(np.dot(input_values, self.weights[error_pair]))

            if self.learning_rate:
                weight_update = np.where(self.freeze_count[error_pair], 0, np.dot(input_values[:, np.newaxis],
                                                                                  - self.neg_error[error_pair][np.newaxis]))
                if error_pair in self.normalize_weights:
                    weight_update -= self.weights[error_pair]*(-input_values[:, np.newaxis] + 1)*self.neg_error_input[error_pair]

                self.weights[error_pair] = np.clip(self.weights[error_pair] + self.learning_rate*weight_update, a_min=0, a_max=None)  # clip weights below 0

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
        self.head_out = np.clip(np.tanh(self.k*self.head), 0, 1)
        if self.log_head_out:
            self.head_out_log.append(self.head_out)

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

        def get_axes_indices(row, column, num_columns):
            if num_columns == 1 or self.num_circuits == 1:
                return max(row, column)
            else:
                return row, column

        if self.log_head or self.log_head_out or self.log_neg_error or self.log_neg_error_out or self.log_pos_error_out:
            fig, ax = plt.subplots(self.num_circuits, self.num_error_pairs, sharex=True, sharey=True, num=self.name + ' Act')
            fig.suptitle("Neural Circuits in Module " + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                for error_pair_num in range(self.num_error_pairs):
                    axes = ax[get_axes_indices(circuit_num, error_pair_num, self.num_error_pairs)]
                    if circuit_num == 0:
                        axes.set_title("Error Pair " + str(error_pair_num), size='medium')
                    if circuit_num == self.num_circuits-1:
                        axes.set_xlabel(x_label)
                    if error_pair_num == 0:
                        axes.set_ylabel("Circuit " + str(circuit_num))
                    if self.log_head:
                        axes.plot(np.array(self.head_log)[:, circuit_num], 'b:', label=r'$' + self.names[circuit_num] + '.h^i$')
                    if self.log_head_out:
                        axes.plot(np.array(self.head_out_log)[:, circuit_num], 'b', label=r'$' + self.names[circuit_num] + '.h$')
                    if self.log_neg_error:
                        axes.plot(np.array(self.neg_error_log)[:, error_pair_num, circuit_num], 'g:',
                                              label=r'$' + self.names[circuit_num] + '.n^i_' + str(error_pair_num) + '$')
                    if self.log_neg_error_out:
                        axes.plot(np.array(self.neg_error_out_log)[:, error_pair_num, circuit_num], 'g',
                                                  label=r'$' + self.names[circuit_num] + '.n_' + str(error_pair_num) + '$')
                    if self.log_pos_error_out:
                        axes.plot(np.array(self.pos_error_out_log)[:, error_pair_num, circuit_num], 'r',
                                              label=r'$' + self.names[circuit_num] + '.p_' + str(error_pair_num) + '$')
                    axes.legend(loc='lower left')

        if self.log_noise_amplitude:
            fig, ax = plt.subplots(self.num_circuits, sharex=True, num=self.name + ' Noise')
            fig.suptitle("Noise Amplitude in Module " + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                ax[circuit_num].plot(np.array(self.noise_amplitude_log)[:, circuit_num])
                ax[circuit_num].set_ylabel("Circuit " + str(circuit_num))
            ax[-1].set_xlabel(x_label)

        if self.log_neg_error_diff:
            fig, ax = plt.subplots(self.num_circuits, self.num_error_pairs, sharex=True, sharey=True, num=self.name + ' Diff')
            fig.suptitle("N(t) - N(t-1) in Module" + self.name, size='large')
            for circuit_num in range(self.num_circuits):
                for error_pair_num in range(self.num_error_pairs):
                    axes = ax[get_axes_indices(circuit_num, error_pair_num, self.num_error_pairs)]
                    if circuit_num == 0:
                        axes.set_title("Error Pair " + str(error_pair_num), size='medium')
                    if circuit_num == self.num_circuits-1:
                        axes.set_xlabel(x_label)
                    if error_pair_num == 0:
                        axes.set_ylabel("Circuit " + str(circuit_num))
                    axes.plot(np.array(self.neg_error_diff_log)[:, error_pair_num, circuit_num], 'r')
                    axes.axhline(self.freeze_threshold, linestyle=':', color='gray')
                    axes.axhline(-self.freeze_threshold, linestyle=':', color='gray')

        if self.log_weights:
            fig, ax = plt.subplots(self.num_circuits, len(self.target_error_pairs), sharex=True, sharey=True, num=self.name + ' W')
            fig.suptitle("Incoming Weights into Module " + self.name, size='large')
            for error_pair_num, error_pair in enumerate(self.target_error_pairs):
                num_input_circuits = self.weights[error_pair].shape[0]
                for circuit_num in range(self.num_circuits):
                    axes_indices = get_axes_indices(circuit_num, error_pair_num, len(self.target_error_pairs))
                    if circuit_num == 0:
                        ax[axes_indices].set_title("Error Pair " + str(error_pair), size='medium')
                    if circuit_num == self.num_circuits - 1:
                        ax[axes_indices].set_xlabel(x_label)
                    if error_pair_num == 0:
                        ax[axes_indices].set_ylabel("Circuit " + str(circuit_num))
                    for input_num in range(num_input_circuits):
                        label = r'$W_{' + self.input_names[error_pair][input_num] + r'.h\rightarrow ' + self.names[circuit_num] + '.n_' + str(error_pair) + '}$'
                        ax[axes_indices].plot(np.array(self.weights_log[error_pair])[:, input_num, circuit_num], label=label)
                    ax[axes_indices].legend(loc='lower right')

        if show:
            plt.show()


class ConvNet:
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.neuron_groups = []
        self.num_groups = 0

    def stack_layer(self, name, num_features, kernel_height, kernel_width, stride,
                  pos_error_to_head=(1, 0), neg_error_to_head=(0.5, 0), time_constant=20,  learning_rate=0.01,
                  noise_max_amplitude=0.15, noise_rise_rate=0.0000002, noise_fall_rate=0.0002, noise_fall_threshold=0.5,
                  dendrite_threshold=2/3, freeze_threshold=0.02,  log_head=False, log_head_out=True,
                  log_neg_error=False, log_neg_error_diff=False, log_neg_error_out=True, log_pos_error_out=True,
                  log_weights=False, log_noise_amplitude=False):

        self.num_groups += 1
        if len(self.neuron_groups) == 0:
            input_height = self.image_height
            input_width = self.image_width
        else:
            input_height = len(self.neuron_groups[-1])
            input_width = len(self.neuron_groups[-1][-1])

        y_in = 0
        y_out = 0
        neuron_groups = []
        while y_in <= input_height - kernel_height:
            x_in = 0
            x_out = 0
            new_row = []
            while x_in <= input_width - kernel_width:
                group_name = name + "_" + str(y_out) + str(x_out)
                new_group = NeuronGroup(group_name, num_features, num_error_pairs=2, pos_error_to_head=pos_error_to_head,
                                        neg_error_to_head=neg_error_to_head, time_constant=time_constant,
                                        learning_rate=learning_rate, noise_max_amplitude=noise_max_amplitude,
                                        noise_rise_rate=noise_rise_rate, noise_fall_rate=noise_fall_rate,
                                        noise_fall_threshold=noise_fall_threshold,  dendrite_threshold=dendrite_threshold,
                                        freeze_threshold=freeze_threshold, log_head=log_head, log_head_out=log_head_out,
                                        log_neg_error=log_neg_error, log_neg_error_diff=log_neg_error_diff,
                                        log_neg_error_out=log_neg_error_out, log_pos_error_out=log_pos_error_out,
                                        log_weights=log_weights, log_noise_amplitude=log_noise_amplitude)
                new_row.append(new_group)
                if len(self.neuron_groups) != 0:
                    for yy_in in range(y_in, y_in + kernel_height):
                        for xx_in in range(x_in, x_in + kernel_width):
                            new_group.enable_connections([self.neuron_groups[-1][yy_in][xx_in]], 0)
                            self.neuron_groups[-1][yy_in][xx_in].enable_connections([new_group], 1)
                x_in += stride
                x_out += 1
            neuron_groups.append(new_row)
            y_in += stride
            y_out += 1
        self.neuron_groups.append(neuron_groups)

    def initialize(self):
        for neuron_groups in self.neuron_groups:
            for row_of_groups in neuron_groups:
                for group in row_of_groups:
                    group.initialize_weights()

    def black_and_white(self, input_image):
        external_input = np.zeros((self.image_height, self.image_width, 2, 2))
        external_input[:, :, 0, 0] = input_image
        external_input[:, :, 0, 1] = 1 - input_image
        return external_input

    def run(self, input_image, simulation_steps=1):
        external_input = self.black_and_white(input_image)
        for step_num in range(simulation_steps):
            for group_num, neuron_groups in enumerate(self.neuron_groups):
                for row_num, row_of_groups in enumerate(neuron_groups):
                    for col_num, group in enumerate(row_of_groups):
                        if group_num == 0:
                            group.step(external_input[row_num, col_num])
                        else:
                            group.step()

    def plot(self):
        reds = colors.LinearSegmentedColormap.from_list('reds', [(1, 0, 0, 0), (1, 0, 0, 1)], N=100)
        greens = colors.LinearSegmentedColormap.from_list('greens', [(0, 1, 0, 0), (0, 1, 0, 1)], N=100)
        blues = colors.LinearSegmentedColormap.from_list('blues', [(0, 0, 1, 0), (0, 0, 1, 1)], N=100)

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
            ax[layer_num].grid(which='minor', linestyle='solid')

        plt.show()