import numpy as np
from os import listdir
from mismatch import ConvolutionalNet
from parameters import operations, macropixels, symbols, pixels

train = 1
test = 1
weights_folder_name = 'weights/operations'

image_size = (70, 10)

net = ConvolutionalNet(image_size[0], image_size[1])
net.stack_layer('p', group_parameters=pixels, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', group_parameters=macropixels, num_features=2, kernel_size=(2, 2), stride=(1, 1))
net.stack_layer('s', group_parameters=symbols, num_features=6, kernel_size=(11, 7), stride=(14, 9), offset=(1, 1))
net.stack_layer('o', group_parameters=operations, num_features=3, kernel_size=(5, 1), stride=(5, 1), terminal=True)
net.initialize_weights()
net.share_weights()

symbols = {}
symbols_folder_name = './../symbols/math 14x10'
for symbol in listdir(symbols_folder_name):
    symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)

training_examples = [['0', '+', '0', '=', '0'], ['1', '+', '0', '=', '1']]
training_rounds = 1

training_steps = 3000
activation_steps = 100

if train:
    net.load_weights(weights_folder_name, ((0, (0, 1)), (1, (0, 1)), (2, (0,))))
    net.learning_off(((0, (0, 1)), (1, (0, 1)), (2, (0,))))
    net.noise_off((0, 1, 2))
    for training_round in range(training_rounds):
        for training_example in training_examples:
            print(f"training {training_example}")
            input_image = symbols[training_example[0]]
            for symbol in training_example[1:]:
                input_image = np.append(input_image, symbols[symbol], 0)
            net.run((activation_steps, activation_steps*2, training_steps, training_steps), input_image)

            input_image = np.zeros(input_image.shape)
            net.run((200, 200, 200, 200), input_image)
            #net.plot(plot_weights=True, show=True)
    net.save_weights(weights_folder_name)

if test:
    net.load_weights(weights_folder_name)
    net.learning_off(((0, (0, 1)), (1, (0, 1)), (2, (0, 1)), (3, (0,))))
    net.noise_off((0, 1, 2, 3))

    test_examples = [['0', '+', '0e', '=', '1']]
    blank_image = np.zeros(image_size)
    num_steps = 80

    forwards = 1
    backwards = 0

    for test_example in test_examples:
        if forwards:
            input_image = blank_image
            for symbol_num, symbol in enumerate(test_example):
                input_image[symbol_num*14:(symbol_num + 1)*14, :] = symbols[symbol]
                net.run((num_steps, num_steps, num_steps, num_steps), input_image)

        else:  # backwards
            input_image = blank_image
            for symbol_num in range(4, -1, -1):
                input_image[symbol_num * 14:(symbol_num + 1) * 14] = symbols[test_example[symbol_num]]
                net.run((num_steps, num_steps, num_steps, num_steps), input_image)

        net.run((num_steps, num_steps, num_steps, num_steps), input_image)
        net.plot(plot_weights=False, show=False)

        print("top down")
        net.set_error_pair_drives([0, 1])
        net.run((num_steps, num_steps*2, num_steps*2, num_steps*2), input_image)
        print("plotting")
        net.plot(plot_weights=False, show=True)

