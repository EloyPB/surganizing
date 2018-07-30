import numpy as np
from surganizing2 import Network


net = Network()
net.add_group("a", 5)
net.add_group("b", 3)

net.build()
net.connect('a', 'b', 0)
net.connect('b', 'a', 1)
net.initialize()

external_input = np.zeros((2, 8))
external_input[0, 0] = 1

net.run(3000, external_input)

external_input = np.zeros((2, 8))
external_input[0, 1] = 1
net.run(3000, external_input)

net.plot_traces()
