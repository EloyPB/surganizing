from Module import Module
import numpy as np

sizes = [3, 4]
s_pairs = [2, 1]
learning_rate = 0.01
tau = 50

A = Module(sizes[0], s_pairs[0], learning_rate, tau)
B = Module(sizes[1], s_pairs[1], learning_rate, tau)

A.enable_connections(to_s_pair=1, from_modules=[B])
A.initialize_weights()

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]
head_input = [np.zeros(size) for size in sizes]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

s_input[0][0, 0] = 1
head_input[1][1] = 0

for t in range(1000):
    for module_num, module in enumerate(modules):
        module.step(head_input[module_num], s_input[module_num])

for module in modules:
    print(module.weights)
    module.plot_circuits(show=module is modules[-1])
