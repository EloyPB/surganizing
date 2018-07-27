import numpy as np
from surganizing2 import Network


net = Network()
net.noise_max_amplitude = 0
net.add_group("a", 5)
net.add_group("b", 3)

net.build()
net.connect('a', 'b', 1)
net.connect('b', 'a', 1)
net.initialize()

external_input = np.zeros((2, 8))
external_input[0, 0] = 1
external_input[0, 5] = 1

net.run(500, external_input)

external_input = np.zeros((2, 8))
net.run(100, external_input)

net.plot_traces()
