# import time
# from os import listdir
# import numpy as np
# from surganizing1b import ConvNet
#
# learn = False
#
# net = ConvNet(28, 10)
# net.stack_layer('m', num_features=2, kernel_height=1, kernel_width=1, stride_y=1, stride_x=1, field=(14, 10))
# net.stack_layer('s', num_features=3, kernel_height=14, kernel_width=10, stride_y=14, stride_x=10, dendrite_threshold=3/4)
# net.initialize()
# net.share_weights()
#
# symbols = {}
# symbols_folder_name = 'symbolss'
# for symbol in listdir(symbols_folder_name):
#     symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)
#
#
# training_rounds = 1
#
# if learn:
#     for training_round in range(training_rounds):
#         for symbol, symbol_image in symbols.items():
#             input_image = np.zeros((28, 10))
#             input_image[0:14, 0:10] = symbol_image
#             net.run(input_image, simulation_steps=3000, layers=2)
#     net.save_weights('weights')
# else:
#     net.load_weights('weights')
#
# input_image = np.zeros((28, 10))
# input_image[0:14, 0:10] = symbols['2']
# net.learning_off(((0, (0, 1)), (1, (0,))))
# net.run(input_image, 200, 2)
#
# print("plotting...")
# net.plot(show=True, plot_weights=True)

import time
import numpy as np
from surganizing1b import ConvNet

learn = False

image_shape = (10, 3)
net = ConvNet(image_shape[0], image_shape[1])
net.stack_layer('m', num_features=2, kernel_height=1, kernel_width=1, stride_y=1, stride_x=1, field=(10, 3), freeze_threshold=100)
net.stack_layer('s', num_features=4, kernel_height=10, kernel_width=3, stride_y=10, stride_x=3, freeze_threshold=100)
net.initialize()
net.share_weights()

zero = np.loadtxt('symbols/0')
two = np.loadtxt('symbols/2')

if learn:
    start_time = time.time()
    input_image = np.zeros(image_shape)
    input_image[0:10, 0:3] = zero[0:10, 0:3]
    net.run(input_image, 8000, 2)
    # input_image = np.zeros(image_shape)
    # input_image[0:3, 0:3] = two
    # net.run(input_image, 3000, 2)
    print(time.time() - start_time)
    net.save_weights('weights')
else:
    net.load_weights('weights')

input_image = np.zeros(image_shape)
input_image[0:10, 0:3] = zero[0:10, 0:3]
#input_image[3:] = two
net.learning_off(((0, (0, 1)), (1, (0,))))
net.run(input_image, 100, 2)
net.run(input_image, 100, 2)

print("plotting...")
net.plot(show=True, plot_weights=True)
