import time
import numpy as np
from os import listdir
from surganizing import ConvolutionalNet
from parameters import operations_p, maps, operations_s


train = 0
test = 1
weights_folder_name = 'weights/operations'

image_size = (70, 10)

net = ConvolutionalNet(image_size[0], image_size[1])
net.stack_layer('p', group_parameters=operations_p, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', group_parameters=maps, num_features=2, kernel_size=(2, 2), stride=(1, 1))
net.stack_layer('s', group_parameters=operations_s, num_features=6, kernel_size=(11, 7), stride=(14, 10), offset=(1, 1), log_head=True)
net.stack_layer('o', group_parameters=maps, num_features=3, kernel_size=(5, 1), stride=(5, 1))
net.initialize()
net.share_weights()

symbols = {}
symbols_folder_name = 'symbols/math 14x10'
for symbol in listdir(symbols_folder_name):
    symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)

training_examples = [['0', '+', '0', '=', '0'], ['1', '+', '0', '=', '1'], ['1', '+', '1', '=', '2']]
training_rounds = 1

if train:
    net.load_weights(weights_folder_name, ((0, (0, 1)), (1, (0, 1)), (2, (0,))))
    net.learning_off(((0, (0, 1)), (1, (0, 1)), (2, (0,))))
    for training_round in range(training_rounds):
        for training_example in training_examples:
            input_image = symbols[training_example[0]]
            for symbol in training_example[1:]:
                input_image = np.append(input_image, symbols[symbol], 0)
            net.run(input_image, simulation_steps=3500, layers=4)
            #net.plot(plot_weights=True, show=True)
    net.save_weights(weights_folder_name)

if test:
    net.load_weights(weights_folder_name)
    net.learning_off(((0, (0, 1)), (1, (0, 1)), (2, (0, 1)), (3, (0, 1))))

    test_examples = [['1', '+', '0', '=', '1']]
    blank_image = np.zeros(image_size)
    num_steps = 200

    static = True
    forwards = True
    backwards = False

    for test_example in test_examples:
        if static:
            input_image = symbols[test_example[0]]
            for symbol in test_example[1:]:
                input_image = np.append(input_image, symbols[symbol], 0)
            net.run(input_image, simulation_steps=100)

        elif forwards:
            input_image = blank_image
            for symbol_num, symbol in enumerate(test_example):
                input_image[symbol_num*14:(symbol_num + 1)*14, :] = symbols[symbol]
                net.run(input_image, simulation_steps=100)
                # net.plot(show=True)

        elif backwards:
            input_image = blank_image
            for symbol_num in range(4, -1, -1):
                input_image[symbol_num * 14:(symbol_num + 1) * 14] = symbols[test_example[symbol_num]]
                net.run(input_image, simulation_steps=100)
                # net.plot(show=True)

        net.run(input_image, simulation_steps=num_steps)
        print("plotting...")
        net.plot(plot_weights=True, show=True)



