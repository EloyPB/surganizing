from Module import Module
import numpy as np
import math

sizes = [18, 3]
s_pairs = [2, 1]
s_pair_weights = [[1.2, 0], [1.2]]
time_constant = 20
max_noise_amplitude = 0.15

log_noise_amplitude = False
log_weights = True
log_s_diff = False

A = Module("A", sizes[0], s_pairs[0], s_pair_weights[0], time_constant, 0,
           log_noise_amplitude=log_noise_amplitude, log_weights=log_weights,
           WTA=False, log_s_diff=log_s_diff)
B = Module("B", sizes[1], s_pairs[1], s_pair_weights[1], time_constant,  max_noise_amplitude,
           log_noise_amplitude=log_noise_amplitude, log_weights=log_weights, log_s_diff=log_s_diff)

A.enable_connections(to_s_pair=1, from_modules=[B])
A.initialize_weights()

B.enable_connections(to_s_pair=0, from_modules=[A])
B.initialize_weights()

modules = [A, B]

training_images = [[1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]]
test_images = [[1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1]]
side_size = int(math.sqrt(len(training_images[0])))


def unfold_image(image):
    pattern = image[:]
    for pixel in image:
        pattern.append(1 - pixel)
    return pattern


training_patterns = [unfold_image(image) for image in training_images]
test_patterns = [unfold_image(image) for image in test_images]

for _ in range(3):
    for training_pattern in training_patterns:
        s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
        for t in range(100):
            for module_num, module in enumerate(modules):
                module.step(s_input[module_num], learning_rate=0.001)
        s_input[0][0] = training_pattern
        for t in range(2000):
            for module_num, module in enumerate(modules):
                module.step(s_input[module_num], learning_rate=0.001)

for test_num, test_pattern in enumerate(test_patterns):
    s_input = [np.zeros((num_s_pairs, size)) for num_s_pairs, size in zip(s_pairs, sizes)]
    for t in range(100):
        for module_num, module in enumerate(modules):
            module.step(s_input[module_num], learning_rate=0)
    s_input[0][0] = test_pattern
    for t in range(100):
        for module_num, module in enumerate(modules):
            module.step(s_input[module_num], learning_rate=0)
    A.plot_image(test_images[test_num], side_size)



print("plotting...")
for module in modules:
    module.plot_circuits(show=module is modules[-1])

