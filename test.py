from Module import Module
import numpy as np

size = 3
module = Module(size, 50)

predictions = np.array([[1, 0, 0], [0, 0, 0]])
head_input = np.array([0, 0, 0])

for i in range(500):
    module.step(predictions, head_input)

predictions = np.array([[0, 0, 0], [0, 0, 0]])
for i in range(500):
    module.step(predictions, head_input)

head_input = np.array([0, 1, 0.5])
for i in range(500):
    module.step(predictions, head_input)

#module.plot_heads()
module.plot_circuits()