import time
import numpy as np
from old_code.surganizing1 import NeuronGroup


sizes = [4, 4]
s_pairs = [2, 1]
s_pair_weights = [[1.2, 0], [1.2]]
time_constant = 20
max_noise_amplitude = 0.15

log_amplitudes = True
log_noise_amplitude = False
log_weights = True
log_s_diff = False

A = NeuronGroup("A", sizes[0], s_pairs[0], s_pair_weights[0], time_constant, max_noise_amplitude,
                log_h_out=log_amplitudes, log_s_out=log_amplitudes, log_sn_out=log_amplitudes,
                log_noise_amplitude=log_noise_amplitude, log_weights=log_weights, log_s_diff=log_s_diff)
B = NeuronGroup("B", sizes[1], s_pairs[1], s_pair_weights[1], time_constant,  max_noise_amplitude,
                log_h_out=log_amplitudes, log_s_out=log_amplitudes, log_sn_out=log_amplitudes,
                log_noise_amplitude=log_noise_amplitude, log_weights=log_weights, log_s_diff=log_s_diff)

A.enable_connections(to_s_pair=1, from_modules=[B])
A.initialize_weights()

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

start_time = time.time()
for input_num in range(2*sizes[0]):
    input_num = input_num%sizes[0]
    s_input[0][0, input_num] = 1
    for t in range(3000):
        for module_num, module in enumerate(modules):
            module.step(s_input[module_num], learning_rate=0.01)
    s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
    for t in range(100):
        for module_num, module in enumerate(modules):
            module.step(s_input[module_num], learning_rate=0.01)

print(time.time() - start_time)
print("plotting...")

for module in modules:
    module.plot_circuits(show=module is modules[-1])
