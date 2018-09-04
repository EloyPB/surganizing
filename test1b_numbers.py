import time
import numpy as np
from os import listdir
from surganizing1b import ConvNet


train_layers = [0, 0, 0]
test = 1
weights_folder_name = 'weights_hardcoded'

image_size = (70, 10)

net = ConvNet(image_size[0], image_size[1])
net.stack_layer('pixels', num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('macropixels', num_features=2, kernel_size=(2, 2), stride=(1, 1), learning_rate=0.0005)
net.stack_layer('symbols', num_features=6, kernel_size=(10, 6), stride=(14, 10), learning_rate=0.0005)
net.stack_layer('operations', num_features=3, kernel_size=(5, 1), stride=(5, 1))
net.initialize()
net.share_weights()

symbols = {}
symbols_folder_name = 'symbols'
for symbol in listdir(symbols_folder_name):
    symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)

training_examples = [['0', '+', '0', '=', '0'], ['1', '+', '1', '=', '2']]
training_rounds = 1

if train_layers[0]:
    for training_round in range(training_rounds):
        input_image = np.zeros(image_size)
        net.run(input_image, simulation_steps=3000, layers=2)
        input_image = np.ones(image_size)
        net.run(input_image, simulation_steps=3000, layers=2)
        net.save_weights(weights_folder_name)

if train_layers[1]:
    net.load_weights(weights_folder_name)
    net.learning_off(((0, (0, 1)), (1, (0,))))
    for training_round in range(training_rounds):
        for symbol, symbol_image in symbols.items():
            input_image = np.zeros(image_size)
            input_image[0:14, 0:10] = symbol_image
            net.run(input_image, simulation_steps=3000, layers=3)
    net.save_weights(weights_folder_name)
    net.plot(plot_weights=True, show=True)

if train_layers[2]:
    net.load_weights(weights_folder_name)
    for training_round in range(training_rounds):
        for training_example in training_examples:
            input_image = symbols[training_example[0]]
            for symbol in training_example[1:]:
                input_image = np.append(input_image, symbols[symbol], 0)
            net.run(input_image, simulation_steps=3000, layers=4)
    net.save_weights(weights_folder_name)

if test:
    net.load_weights(weights_folder_name)
    net.learning_off(((0, (0, 1)), (1, (0, 1)), (2, (0, 1)), (3, (0, 1))))

    test_examples = [['2', '+', '0', '=', '2']]
    blank_image = np.zeros(image_size)
    num_steps = 100

    for test_example in test_examples:
        input_image = symbols[test_example[0]]
        for symbol in test_example[1:]:
            input_image = np.append(input_image, symbols[symbol], 0)

        start_time = time.time()
        net.run(blank_image, simulation_steps=1)
        net.run(input_image, simulation_steps=num_steps)
        net.run(input_image)
        print(time.time() - start_time)
        print("plotting...")
        net.plot(plot_weights=True, show=True)



