import time
import numpy as np
from mismatch import ConvolutionalNet
from parameters import pixels, macropixels

learn = True

net = ConvolutionalNet(10, 5)
net.stack_layer('p', pixels, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', macropixels, num_features=2, kernel_size=(2, 2), stride=(1, 1), terminal=True)
net.initialize_weights()
net.share_weights()

input_image = np.zeros((10, 5))

if learn:
    start_time = time.time()
    net.run((3000, 3000), input_image)
    input_image = np.ones((10, 5))
    net.run((3000, 3000), input_image)
    print(time.time() - start_time)
    net.save_weights('weights/maps_line')
else:
    net.load_weights('weights/maps_line')

net.learning_off(((0, (0, 1)), (1, (0,))))
input_image = np.zeros((10, 5))
input_image[4, :] = 1
net.run((200, 200), input_image)

print("plotting...")
net.plot(show=True, plot_weights=True)
