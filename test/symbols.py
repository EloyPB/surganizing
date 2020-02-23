"""Learn symbols in a three layered convolutional network."""


import numpy as np
from os import listdir
from mismatch import ConvolutionalNet
from parameters import pixels, macropixels, symbols

learn = [1, 1]
training_rounds = 1
weights_folder = 'weights/operations'
symbols_to_learn = ['5', '6', '7', '8', '0', '1', '2', '3', '4', '=', '+']
test = 1

image_shape = (14, 10)
net = ConvolutionalNet(image_shape[0], image_shape[1])
net.stack_layer('p', group_parameters=pixels, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', group_parameters=macropixels, num_features=2, kernel_size=(2, 2), stride=(1, 1))
# net.stack_layer('s', group_parameters=symbols, num_features=6, kernel_size=(11, 7), stride=(13, 9), offset=(1, 1))
net.stack_layer('s', group_parameters=symbols, num_features=12, kernel_size=(11, 7), stride=(14, 10), offset=(1, 1))
net.initialize_weights()
net.share_weights()

symbols = {}
symbols_folder_name = './../symbols/math 14x10'
for symbol in listdir(symbols_folder_name):
    symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)

training_steps = 3000
activation_steps = 100

if learn[0]:
    net.noise_off((0,))
    net.learning_off(((1, (1,)),))
    for training_round in range(training_rounds):
        print("training black")
        input_image = np.zeros(image_shape)
        net.run(input_image=input_image, simulation_steps=(training_steps, training_steps))
        print("training white")
        input_image = np.ones(image_shape)
        net.run(input_image=input_image, simulation_steps=(training_steps, training_steps))
        net.save_weights(weights_folder)

if learn[1]:
    net.noise_off((0, 1))
    net.learning_on(((1, (1,)),))
    net.learning_off(((0, (0, 1)), (1, (0,))))
    net.load_weights(weights_folder)
    for training_round in range(training_rounds):
        for symbol_to_learn in symbols_to_learn:
            print(f"training round: {training_round}, symbol to learn: {symbol_to_learn}")
            input_image = symbols[symbol_to_learn]
            net.run(input_image=input_image, simulation_steps=(activation_steps, training_steps, training_steps))
            print(net.neuron_groups[2][0][0].noise_amplitude)
    net.save_weights(weights_folder)

if test:
    net.load_weights(weights_folder)
    net.learning_off(((0, (0, 1)), (1, (0, 1), (2, (0,)))))
    for symbol_to_learn in symbols_to_learn:
        input_image = symbols[symbol_to_learn]
        net.run(input_image=input_image, simulation_steps=(activation_steps, activation_steps*2, activation_steps*3))
        net.plot(show=True, plot_weights=True)

