import time
import numpy as np
from old_code.surganizing2 import Network


net = Network()
net.add_group("a", 4)
net.add_group("b", 4)

net.build()
net.connect('a', 'b', 0)
net.connect('b', 'a', 1)
net.initialize()

start_time = time.time()
for i in range(4):
    external_input = np.zeros((2, 8))
    external_input[0, i] = 1
    net.run(3000, external_input)
    external_input = np.zeros((2, 8))
    net.run(100, external_input)
print(time.time() - start_time)

net.plot_traces()