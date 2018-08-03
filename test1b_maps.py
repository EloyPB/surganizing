import time
import numpy as np
from surganizing1b import ConvNet

net = ConvNet(70, 14)
net.stack_layer('pixels', 2, 1, 1, 1)
net.stack_layer('macropixels', 2, 2, 2, 1)
net.stack_layer('symbols', 10, 13, 13, 13)
net.stack_layer('operations', 5, 5, 1, 5)
net.initialize()

for layer in net.neuron_groups:
    print(len(layer), len(layer[0]))

input_image = np.zeros((70, 14))
input_image[10, :] = 1

start_time = time.time()
net.run(input_image, 50)
print(time.time() - start_time)

print("plotting...")
net.plot()