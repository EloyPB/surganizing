import time
import numpy as np
from surganizing1b import ConvNet

net = ConvNet(70, 14)
net.stack_layer('pixels', num_features=2, kernel_height=1, kernel_width=1, stride_y=1, stride_x=1, field=(1, 1))
net.stack_layer('macropixels', num_features=2, kernel_height=2, kernel_width=2, stride_y=1, stride_x=1, field=(14, 14))
net.stack_layer('symbols', num_features=10, kernel_height=14, kernel_width=14, stride_y=14, stride_x=14, field=(5, 1))
net.stack_layer('operations', num_features=2, kernel_height=5, kernel_width=1, stride_y=5, stride_x=1)
net.initialize()
net.share_weights()

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