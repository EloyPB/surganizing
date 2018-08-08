import time
import numpy as np
from surganizing1b import ConvNet

learn = False

net = ConvNet(10, 5)
net.stack_layer('p', 2, 1, 1, 1, learning_rate=0.001)
net.stack_layer('m', 2, 2, 2, 1, learning_rate=0.001)
net.initialize()
net.share_weights()

input_image = np.zeros((10, 5))

if learn:
    start_time = time.time()
    net.run(input_image, 2, 3000)
    print(time.time() - start_time)
    net.save_weights()
else:
    net.load_weights()

net.learning_off()
input_image = np.zeros((10, 5))
input_image[4, :] = 1
net.run(input_image, 2, 200)

print("plotting...")
net.plot(show=True)

# net.neuron_groups[0][0][0].plot_circuits()
# net.neuron_groups[1][0][0].plot_circuits(show=True)
