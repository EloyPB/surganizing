import numpy as np
from surganizing1 import FeatureMaps
from surganizing1 import ConvNet

learning_rate = 0.005

first_map = FeatureMaps("first")
first_map.build_initial(height=4, width=4, num_features=2)

second_map = FeatureMaps("second")
second_map.build_on_top(input_feature_maps=first_map, kernel_height=2, kernel_width=2, stride=1, num_features=2)

net = ConvNet([first_map, second_map])

input_pattern = np.zeros((4, 4, 2))
input_pattern[:, :, 0] = 1

for t in range(2000):
    print(t)
    net.run_step(input_pattern, [learning_rate, learning_rate])

net.plot_activities()

input_pattern = np.zeros((4, 4, 2))
input_pattern[:, :, 1] = 1

for t in range(2000):
    print(t)
    net.run_step(input_pattern, [learning_rate, learning_rate])

net.plot_activities()

input_pattern = np.zeros((4, 4, 2))
input_pattern[:2, :, 0] = 1
input_pattern[2:, :, 1] = 1

for t in range(200):
    print(t)
    net.run_step(input_pattern, [0, 0])

net.plot_activities()

print("done")