import time
import numpy as np
from surganizing import CircuitGroup
import parameters.basic as parameters


sizes = [2, 2]
num_error_pairs = [2, 1]
weight_normalizing_pairs = [[0], [0]]

log_amplitudes = True
log_noise_amplitude = False
log_weights = True

A = CircuitGroup(name="A", parameters=parameters, num_circuits=sizes[0], num_error_pairs=num_error_pairs[0],
                 weight_normalizing_pairs=weight_normalizing_pairs[0])

B = CircuitGroup(name="B", parameters=parameters, num_circuits=sizes[1], num_error_pairs=num_error_pairs[1],
                 weight_normalizing_pairs=weight_normalizing_pairs[1])

A.enable_connections(input_groups=[B], target_error_pair=1)
A.set_error_pair_drives([1, 0])
A.initialize_weights()

B.enable_connections(input_groups=[A], target_error_pair=0)
B.set_error_pair_drives([1])
B.initialize_weights()

groups = [A, B]
external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]

start_time = time.time()

external_input[0][0, 0] = 1
for t in range(3000):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
for t in range(100):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

external_input[0][0, 0] = 1
for t in range(200):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

print(time.time() - start_time)
print("plotting...")

for group in groups:
    group.plot_circuits(show=group is groups[-1])
