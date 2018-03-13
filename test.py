from Module import Module
import numpy as np

sizes = [3, 3]
s_pairs = [2, 1]
s_pair_weights = [[1.2, 0], [1.2]]
learning_rate = 0.005
tau = 50

log_noise_amplitude = True
log_weights = True
log_s_diff = False

A = Module("A", sizes[0], s_pairs[0], s_pair_weights[0], learning_rate, tau, 0.15,
           log_noise_amplitude=log_noise_amplitude, log_weights=log_weights, log_s_diff=log_s_diff)
B = Module("B", sizes[1], s_pairs[1], s_pair_weights[1], learning_rate, tau, 0.15,
           log_noise_amplitude=log_noise_amplitude, log_weights=log_weights, log_s_diff=log_s_diff)

A.enable_connections(to_s_pair=1, from_modules=[B])
A.initialize_weights()

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]
head_input = [np.zeros(size) for size in sizes]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

for input_num in range(2*sizes[0]):
    input_num = input_num%sizes[0]
    s_input[0][0, input_num] = 1
    for t in range(3000):
        for module_num, module in enumerate(modules):
            module.step(head_input[module_num], s_input[module_num])
    s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
    for t in range(1000):
        for module_num, module in enumerate(modules):
            module.step(head_input[module_num], s_input[module_num])


print("plotting...")
for module in modules:
    # print(module.weights)
    module.plot_circuits(show=module is modules[-1])
