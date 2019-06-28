import numpy as np
from old.surganizing1c import ConvNet


net = ConvNet(3, 3)
net.stack_layer('p', num_features=2, kernel_size=(1, 1), stride=(1, 1), log_head=True, log_neg_error=True)
net.stack_layer('s', num_features=1, kernel_size=(3, 3), stride=(3, 3), log_head=True, log_neg_error=True)
net.initialize()

one = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
two = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]])
three = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1]])

input_images = [one, two, three]

for iteration in range(1):
    net.run(input_images[iteration % len(input_images)], 1000, 1)


print("plotting...")
net.neuron_groups[0][0][0].plot_circuits()
net.neuron_groups[1][0][0].plot_circuits()
net.plot(show=True, plot_weights=True)
