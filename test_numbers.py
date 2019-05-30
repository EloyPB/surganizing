import numpy as np
from os import listdir
from surganizing import ConvolutionalNet
from parameters import operations_p, maps, operations_s

learn = [0, 0]
training_rounds = 1
weights_folder = 'weights/numbers'
symbols_to_learn = ['0', '1', '2', '=', '+']
test = 1

image_shape = (14, 10)
net = ConvolutionalNet(image_shape[0], image_shape[1])
net.stack_layer('p', group_parameters=operations_p, num_features=2, kernel_size=(1, 1), stride=(1, 1))
net.stack_layer('m', group_parameters=maps, num_features=2, kernel_size=(2, 2), stride=(1, 1))
net.stack_layer('s', group_parameters=operations_s, num_features=6, kernel_size=(11, 7), stride=(14, 10), offset=(1, 1))
net.initialize()
net.share_weights()

symbols = {}
symbols_folder_name = 'symbols/math 14x10'
for symbol in listdir(symbols_folder_name):
    symbols[symbol] = np.loadtxt(symbols_folder_name + '/' + symbol)


if learn[0]:
    for training_round in range(training_rounds):
        input_image = np.zeros(image_shape)
        net.run(input_image=input_image, simulation_steps=4000, layers=2)
        #net.plot(plot_weights=True, show=True)
        input_image = np.ones(image_shape)
        net.run(input_image=input_image, simulation_steps=4000, layers=2)
        #net.plot(plot_weights=True, show=True)
        net.save_weights(weights_folder)

if learn[1]:
    net.load_weights(weights_folder)
    net.learning_off(((0, (0, 1)), (1, (0,))))
    for training_round in range(training_rounds):
        for symbol_to_learn in symbols_to_learn:
            input_image = symbols[symbol_to_learn]
            net.run(input_image, simulation_steps=3000, layers=3)
    net.save_weights(weights_folder)

if test:
    net.load_weights(weights_folder)
    net.learning_off(((0, (0, 1)), (1, (0, 1), (2, (0,)))))
    for symbol_to_learn in symbols_to_learn:
        input_image = symbols[symbol_to_learn]
        net.run(input_image, simulation_steps=300, layers=3)
        net.plot(show=True, plot_weights=True)

