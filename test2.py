import numpy as np
from surganizing2 import Network


net = Network()
net.add_group("a", 5)
net.add_group("b", 3)

net.initialize_activities()

external_input = np.zeros((2, 8))
external_input[0, 3] = 1
net.run(100, external_input)

external_input = np.zeros((2, 8))
net.run(100, external_input)

net.plot_traces()
