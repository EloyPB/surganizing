"""Two layered convolutional network learns two simple symbols."""


import time
import numpy as np
from mismatch import ConvolutionalNet
import parameters.basic as parameters

learn = True

net = ConvolutionalNet(6, 3)
net.stack_layer('m', parameters, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('s', parameters, num_features=4, kernel_size=(3, 3), stride=(3, 3))
net.initialize_weights()
net.share_weights()

cross = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

if learn:
    start_time = time.time()
    input_image = np.zeros((6, 3))
    input_image[0:3, 0:3] = cross
    net.run((3000, 3000), input_image)
    input_image = np.zeros((6, 3))
    input_image[0:3, 0:3] = plus
    net.run((3000, 3000), input_image)
    print(time.time() - start_time)
    net.save_weights('weights/maps_cross')
else:
    net.load_weights('weights/maps_cross')

input_image = np.zeros((6, 3))
input_image[0:3, 0:3] = cross
input_image[3:] = plus
net.learning_off(((0, (0, 1)), (1, (0,))))
net.run((200, 200), input_image)

print("plotting...")
net.plot(show=True, plot_weights=True)