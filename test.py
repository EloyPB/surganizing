from Module import Module
import numpy as np

sizes = [3, 2]
s_pairs = [2, 1]
learning_rate = 0.005
tau = 50

A = Module(sizes[0], s_pairs[0], learning_rate, tau)
B = Module(sizes[1], s_pairs[1], learning_rate, tau)

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]
head_input = [np.zeros(size) for size in sizes]
s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]

s_input[0][0, 0] = 1
head_input[1][1] = 1

for t in range(1000):
    for module_num, module in enumerate(modules):
        module.step(head_input[module_num], s_input[module_num])

for module in modules:
    print(module.weights)
    module.plot_circuits(show=module is modules[-1])
