"""Demonstration of the evolution of intrinsic noise levels and the overwriting of the least used unit."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mismatch import CircuitGroup
import parameters.basic as parameters


names = ["A", "B", "C", "D"]
sizes = [1, 1, 1, 2]
num_error_pairs = [2, 2, 2, 1]
feedforward_input_pairs = [[], [], [], [0]]

training_steps = 3000
few_steps = 1000
reset_steps = 200

log_noise_amplitude = True
log_weights = True

circuit_groups = []
for group_num in range(len(names)):
    circuit_groups.append(CircuitGroup(name=names[group_num], parameters=parameters, num_circuits=sizes[group_num],
                                       num_error_pairs=num_error_pairs[group_num],
                                       feedforward_input_pairs=feedforward_input_pairs[group_num],
                                       log_noise_amplitude=log_noise_amplitude, log_weights=log_weights))

for circuit_group in circuit_groups[:-1]:
    circuit_group.enable_connections(input_groups=[circuit_groups[-1]], target_error_pair=1)
    circuit_group.set_error_pair_drives([1, 0])
    circuit_group.redistribution_rate = 0
    circuit_group.noise_max_amplitude = 0
    circuit_group.initialize_weights()

    circuit_groups[-1].enable_connections(input_groups=[circuit_group], target_error_pair=0)
    circuit_groups[-1].set_error_pair_drives([1])
    circuit_groups[-1].initialize_weights()


def run(active_group_num, steps):
    external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
    if active_group_num is not None:
        external_input[active_group_num][0, 0] = 1
    for t in range(steps):
        for group_num, group in enumerate(circuit_groups):
            group.step(external_input=external_input[group_num])


run(active_group_num=0, steps=training_steps)
run(active_group_num=None, steps=reset_steps)
run(active_group_num=1, steps=training_steps)
run(active_group_num=None, steps=reset_steps)

for i in range(8):
    run(active_group_num=i % 2 + 1, steps=few_steps)
    run(active_group_num=None, steps=reset_steps)


print("plotting...")

orange = LinearSegmentedColormap.from_list('orange', [colors.to_rgba('C1', 0), colors.to_rgba('C1', 1)], N=100)
green = LinearSegmentedColormap.from_list('green', [colors.to_rgba('C2', 0), colors.to_rgba('C2', 1)], N=100)
blue = LinearSegmentedColormap.from_list('blue', [colors.to_rgba('C0', 0), colors.to_rgba('C0', 1)], N=100)

fig, axes = plt.subplots(5, 1, sharex='col', gridspec_kw={'height_ratios': (3, 2, 10, 10, 10)})

h = np.empty((sum(sizes), len(circuit_groups[0].head_out_log)))
p = np.empty((sum(sizes), len(circuit_groups[0].head_out_log)))
n = np.empty((sum(sizes), len(circuit_groups[0].head_out_log)))

row = 0
circuit_names = []
for group_num, circuit_group in enumerate(circuit_groups):
    for circuit_num in range(sizes[group_num]):
        h[row] = np.array(circuit_group.head_out_log)[:, circuit_num]
        n[row] = np.array(circuit_group.neg_error_out_log)[:, -1, circuit_num]
        p[row] = np.array(circuit_group.pos_error_out_log)[:, -1, circuit_num]
        circuit_names.append(r"$" + circuit_group.names[circuit_num] + "$")
        row += 1

axes[0].matshow(h[:sum(sizes[:-1])], cmap=blue, aspect="auto", vmin=0, vmax=1)
axes[0].matshow(n[:sum(sizes[:-1])], cmap=green, aspect="auto", vmin=0, vmax=1)
axes[0].matshow(p[:sum(sizes[:-1])], cmap=orange, aspect="auto", vmin=0, vmax=1)
axes[0].set_yticks(range(sum(sizes[:-1])))
axes[0].set_yticklabels(circuit_names[:sum(sizes[:-1])])
axes[0].set_yticks(np.arange(-0.5, sum(sizes[:-1])), minor=True)
axes[0].yaxis.grid(True, which="minor")

axes[1].matshow(h[sum(sizes[:-1]):], cmap=blue, aspect="auto", vmin=0, vmax=1)
axes[1].matshow(n[sum(sizes[:-1]):], cmap=green, aspect="auto", vmin=0, vmax=1)
axes[1].matshow(p[sum(sizes[:-1]):], cmap=orange, aspect="auto", vmin=0, vmax=1)
axes[1].set_yticks(range(sizes[-1]))
axes[1].set_yticklabels(circuit_names[sum(sizes[:-1]):])
axes[1].set_yticks(np.arange(-0.5, sizes[-1]), minor=True)
axes[1].yaxis.grid(True, which="minor")

for circuit_num in range(circuit_groups[-1].num_circuits):
    label = r'$W_{' + circuit_groups[-1].names[circuit_num] + '}$'
    axes[2].plot(np.array(circuit_groups[-1].noise_amplitude_log)[:, circuit_num], label=label)
axes[2].set_ylabel('Intrinsic\nnoise amplitudes')
axes[3].legend(loc="upper right", fontsize="small")

num_input_circuits = circuit_groups[-1].weights[0].shape[0]
for circuit_num in range(circuit_groups[-1].num_circuits):
    for input_num in range(num_input_circuits):
        label = r'$W_{' + circuit_groups[-1].input_names[0][input_num] + r'.h\rightarrow ' \
                + circuit_groups[-1].names[circuit_num] + '.n_' + str(0) + '}$'
        axes[3].plot(np.array(circuit_groups[-1].weights_log[0])[:, input_num, circuit_num], label=label)
axes[3].set_ylim([0, 1.3])
axes[3].legend(loc="upper left", ncol=8, fontsize="small")

for circuit_group in circuit_groups[:-1]:
    for input_num in range(sizes[-1]):
        label = r'$W_{' + circuit_group.input_names[1][input_num] + r'.h\rightarrow ' \
                    + circuit_group.names[0] + '.n_' + str(1) + '}$'
        axes[4].plot(np.array(circuit_group.weights_log[1])[:, input_num, 0], label=label)
axes[4].legend(loc="upper left", ncol=8, fontsize="small")
axes[4].set_xlabel("Simulation steps")
axes[4].set_xlim([0, len(circuit_groups[0].head_out_log)])
axes[4].set_ylim([0, 1.3])

# plt.savefig("demo_noise.pdf")
plt.show()

