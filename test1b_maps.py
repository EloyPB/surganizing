import time
import numpy as np
from surganizing1b import ConvNet

net = ConvNet(10, 10)
net.stack_layer('pixels', 2, 1, 1, 1)
net.stack_layer('macropixels', 2, 2, 2, 1)
net.initialize()
net.share_weights()

print(net.neuron_groups[0][0][0].weights)
net.neuron_groups[0][round(len(net.neuron_groups[0])/2)][round(len(net.neuron_groups[0][0])/2)].weights[1][0,0] = 1
print(net.neuron_groups[0][-1][0].weights)

for layer in net.neuron_groups:
    print(len(layer), len(layer[0]))

input_image = np.zeros((10, 10))
input_image[4, :] = 1

start_time = time.time()
net.run(input_image, 1, 10)
print(time.time() - start_time)

print("plotting...")
net.plot()