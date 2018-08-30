import time
import numpy as np
from surganizing1b import NeuronGroup


sizes = [2, 2, 2]
s_pairs = [2, 1, 1]
neg_error_to_head = [[0.6, 0], [0.6], [0.6]]
pos_error_to_head = [[1.2, 0], [1.2], [1.2]]
normalize_weights = [[0], [0], [0]]
time_constant = 20
max_noise_amplitude = 0.15

log_amplitudes = True
log_noise_amplitude = False
log_weights = True
log_s_diff = False

A = NeuronGroup(name="A", num_circuits=sizes[0], num_error_pairs=s_pairs[0], pos_error_to_head=pos_error_to_head[0],
                neg_error_to_head=neg_error_to_head[0], normalize_weights=normalize_weights[0], dendrite_threshold=0,
                time_constant=time_constant, noise_max_amplitude=max_noise_amplitude)

B = NeuronGroup(name="B", num_circuits=sizes[1], num_error_pairs=s_pairs[1], pos_error_to_head=pos_error_to_head[1],
                neg_error_to_head=neg_error_to_head[1], normalize_weights=normalize_weights[1],  dendrite_threshold=0,
                time_constant=time_constant, noise_max_amplitude=max_noise_amplitude)

C = NeuronGroup(name="C", num_circuits=sizes[2], num_error_pairs=s_pairs[2], pos_error_to_head=pos_error_to_head[2],
                neg_error_to_head=neg_error_to_head[2], normalize_weights=normalize_weights[2], dendrite_threshold=0,
                time_constant=time_constant, noise_max_amplitude=max_noise_amplitude)

A.enable_connections(input_groups=[B, C], target_error_pair=1)
A.initialize_weights()

B.enable_connections(input_groups=[A], target_error_pair=0)
B.initialize_weights()

C.enable_connections(input_groups=[A], target_error_pair=0)
C.initialize_weights()

modules = [A, B, C]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

start_time = time.time()
s_input[0][0, 0] = 1
for t in range(20000):
    for module_num, module in enumerate(modules):
        module.step(external_input=s_input[module_num])


print(time.time() - start_time)
print(A.weights)
print("plotting...")

for module in modules:
    module.plot_circuits(show=module is modules[-1])
