from Module import Module
import numpy as np

sizes = [3, 3]
s_pairs = [2, 1]
s_pair_weights = [[1, 0], [1]]
learning_rate = 0.005
tau = 50

A = Module(sizes[0], s_pairs[0], s_pair_weights[0], learning_rate, tau, 0.05)
B = Module(sizes[1], s_pairs[1], s_pair_weights[1], learning_rate, tau, 0.05)

A.enable_connections(to_s_pair=1, from_modules=[B])
A.initialize_weights()

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]
head_input = [np.zeros(size) for size in sizes]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

for input_num in range(sizes[0]):
    s_input[0][0, input_num] = 1
    for t in range(2000):
        for module_num, module in enumerate(modules):
            module.step(head_input[module_num], s_input[module_num])
    s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
    for t in range(1000):
        for module_num, module in enumerate(modules):
            module.step(head_input[module_num], s_input[module_num])


for module in modules:
    # print(module.weights)
    module.plot_circuits(show=module is modules[-1])
