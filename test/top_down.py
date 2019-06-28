import time
import numpy as np
from mismatch import CircuitGroup
import parameters.basic as parameters


log_amplitudes = True
log_noise_amplitude = False
log_weights = True
log_s_diff = False

num_bottom_groups = 8
groups = []
sizes = []
num_error_pairs = []

top_group = CircuitGroup(name="t", parameters=parameters, num_circuits=2, num_error_pairs=1,
                         feedforward_input_pairs=[0])

for bottom_group_num in range(num_bottom_groups):
    groups.append(CircuitGroup(name="b" + str(bottom_group_num), parameters=parameters, num_circuits=1,
                               num_error_pairs=2, feedforward_input_pairs=[0], log_weights=False))

    groups[-1].set_error_pair_drives([1, 0])
    groups[-1].enable_connections(input_groups=[top_group], target_error_pair=1)
    groups[-1].initialize_weights()

    top_group.enable_connections(input_groups=[groups[-1]], target_error_pair=0)

    sizes.append(1)
    num_error_pairs.append(2)


top_group.set_error_pair_drives([1])
top_group.initialize_weights()

groups.append(top_group)
sizes.append(2)
num_error_pairs.append(1)

external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
for bottom_group_num in range(num_bottom_groups):
    external_input[bottom_group_num][0, 0] = 1

start_time = time.time()
for t in range(3000):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
for t in range(100):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
for bottom_group_num, bottom_group in enumerate(groups[0:num_bottom_groups - 1]):
    external_input[bottom_group_num][0, 0] = 1
for bottom_group in groups[0:num_bottom_groups]:
    bottom_group.learning_off([0, 1])
top_group.learning_off([0])
for t in range(100):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

for bottom_group in groups[0:num_bottom_groups]:
    bottom_group.set_error_pair_drives([0, 1])
for t in range(100):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

print(time.time() - start_time)
print("plotting...")

for group_num, group in enumerate(groups):
    group.plot_circuits(show=group_num == len(groups) - 1)
