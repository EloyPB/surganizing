import time
import numpy as np
from surganizing import ConvolutionalNet
from parameters import pixels, macropixels

learn = True

net = ConvolutionalNet(10, 5)
net.stack_layer('p', pixels, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', macropixels, num_features=2, kernel_size=(2, 2), stride=(1, 1))
net.initialize()
net.share_weights()

input_image = np.zeros((10, 5))

if learn:
    start_time = time.time()
    net.run(input_image, 3000, 2)
    print(time.time() - start_time)
    net.save_weights('weights/maps_line')
else:
    net.load_weights('weights/maps_line')

net.learning_off(((0, (0, 1)), (1, (0,))))
input_image = np.zeros((10, 5))
input_image[4, :] = 1
net.run(input_image, 200, 2)

print("plotting...")
net.plot(show=True, plot_weights=True)
