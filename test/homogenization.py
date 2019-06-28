import time
import numpy as np
from mismatch import CircuitGroup
import parameters.basic as parameters


sizes = [1, 1, 1]
num_error_pairs = [2, 2, 1]
feedforward_input_pairs = [[], [], [0]]

log_noise_amplitude = False
log_weights = True

A = CircuitGroup(name="A", parameters=parameters, num_circuits=sizes[0], num_error_pairs=num_error_pairs[0],
                 feedforward_input_pairs=feedforward_input_pairs[0], log_noise_amplitude=log_noise_amplitude,
                 log_weights=log_weights)

B = CircuitGroup(name="B", parameters=parameters, num_circuits=sizes[1], num_error_pairs=num_error_pairs[1],
                 feedforward_input_pairs=feedforward_input_pairs[1], log_noise_amplitude=log_noise_amplitude,
                 log_weights=log_weights)

C = CircuitGroup(name="C", parameters=parameters, num_circuits=sizes[2], num_error_pairs=num_error_pairs[2],
                 feedforward_input_pairs=feedforward_input_pairs[2], log_noise_amplitude=log_noise_amplitude,
                 log_weights=log_weights)

A.enable_connections(input_groups=[C], target_error_pair=1)
A.set_error_pair_drives([1, 0])
A.noise_amplitude[:] = 0
A.noise_max_amplitude = 0
A.initialize_weights()

B.enable_connections(input_groups=[C], target_error_pair=1)
B.set_error_pair_drives([1, 0])
B.noise_amplitude[:] = 0
B.noise_max_amplitude = 0
B.initialize_weights()

C.enable_connections(input_groups=[A, B], target_error_pair=0)
C.set_error_pair_drives([1])
C.initialize_weights()

groups = [A, B, C]
external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]

start_time = time.time()
external_input[0][0, 0] = 1
for t in range(3000):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])
external_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(num_error_pairs, sizes)]
external_input[0][0, 0] = 1
external_input[1][0, 0] = 1
for t in range(1000):
    for group_num, group in enumerate(groups):
        group.step(external_input=external_input[group_num])

print(time.time() - start_time)
print("plotting...")

for group in groups:
    group.plot_circuits(show=group is groups[-1])
