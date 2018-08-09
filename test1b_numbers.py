import time
import numpy as np
from surganizing1b import ConvNet

net = ConvNet(70, 14)
net.stack_layer('pixels', 2, 1, 1, 1, 1)
net.stack_layer('macropixels', 2, 2, 2, 1, 1)
net.stack_layer('symbols', 10, 14, 14, 14, 14)
net.stack_layer('operations', 2, 5, 1, 5, 1)
net.initialize()

print(net.weight_shapes)

for layer in net.neuron_groups:
    print(len(layer), len(layer[0]))

input_image = np.zeros((70, 14))
input_image[10, :] = 1

start_time = time.time()
net.run(input_image, 4, 10)
print(time.time() - start_time)

print("plotting...")
net.plot(True)