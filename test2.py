import numpy as np
from surganizing2 import Network


net = Network()
net.add_group("a", 5)
net.add_group("b", 3)

net.initialize()

external_input = np.zeros((2, 8))
external_input[0, 0] = 0

net.run(100, external_input)

external_input = np.zeros((2, 8))
net.run(100, external_input)

net.plot_traces()
