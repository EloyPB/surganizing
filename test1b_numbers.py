import time
import numpy as np
from os import listdir
from surganizing1b import ConvNet


train = True

image_size = (70, 14)

net = ConvNet(image_size[0], image_size[1])
net.stack_layer('pixels', num_features=2, kernel_height=1, kernel_width=1, stride_y=1, stride_x=1, field=(1, 1))
net.stack_layer('macropixels', num_features=2, kernel_height=2, kernel_width=2, stride_y=1, stride_x=1, field=(14, 14))
net.stack_layer('symbols', num_features=10, kernel_height=14, kernel_width=14, stride_y=14, stride_x=14, field=(5, 1))
net.stack_layer('operations', num_features=5, kernel_height=5, kernel_width=1, stride_y=5, stride_x=1)
net.initialize()
net.share_weights()

symbols = {}
folder_name = 'symbols'
for symbol in listdir(folder_name):
    symbols[symbol] = np.loadtxt(folder_name + '/' + symbol)

operations = [['0', '+', '0', '=', '0'], ['0', '+', '1', '=', '1'], ['1', '+', '0', '=', '1']]

if train:
    num_rounds = 1
    num_steps = 3000
else:
    num_rounds = 1
    num_steps = 3
    net.load_weights("weights")
    net.learning_off()

blank_image = np.zeros(image_size)
for round_num in range(num_rounds):
    for operation in operations:
        input_image = symbols[operation[0]]
        for symbol in operation[1:]:
            input_image = np.append(input_image, symbols[symbol], 0)

        start_time = time.time()
        net.run(blank_image, simulation_steps=1)
        net.run(input_image, simulation_steps=num_steps)
        print(time.time() - start_time)

if train:
    net.save_weights("weights")

print("plotting...")
net.plot(True)