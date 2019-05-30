import time
import numpy as np
from surganizing1b import CircuitGroup


sizes = [4, 4]
s_pairs = [2, 1]
neg_error_to_head = [[0.6, 0], [0.6]]
pos_error_to_head = [[1.2, 0], [1.2]]
normalize_weights = [[0], [0]]
time_constant = 20
max_noise_amplitude = 0.15

log_amplitudes = True
log_noise_amplitude = False
log_weights = True
log_s_diff = False

A = CircuitGroup(name="A", num_circuits=sizes[0], num_error_pairs=s_pairs[0], pos_error_to_head=pos_error_to_head[0],
                 neg_error_to_head=neg_error_to_head[0], normalize_weights=normalize_weights[0],
                 time_constant=time_constant, noise_max_amplitude=max_noise_amplitude)

B = CircuitGroup(name="B", num_circuits=sizes[1], num_error_pairs=s_pairs[1], pos_error_to_head=pos_error_to_head[1],
                 neg_error_to_head=neg_error_to_head[1], normalize_weights=normalize_weights[1],
                 time_constant=time_constant, noise_max_amplitude=max_noise_amplitude)

A.enable_connections(input_groups=[B], target_error_pair=1)
A.initialize_weights()

B.enable_connections(input_groups=[A], target_error_pair=0)
B.initialize_weights()

modules = [A, B]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

start_time = time.time()
for input_num in range(2*sizes[0]):
    input_num = input_num%sizes[0]
    s_input[0][0, input_num] = 1
    for t in range(3000):
        for module_num, module in enumerate(modules):
            module.step(external_input=s_input[module_num])
    s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
    for t in range(100):
        for module_num, module in enumerate(modules):
            module.step(external_input=s_input[module_num])

print(time.time() - start_time)
print("plotting...")

for module in modules:
    module.plot_circuits(show=module is modules[-1])
