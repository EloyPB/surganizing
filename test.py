#from Surganizing.Module import Module
import numpy as np
from matplotlib import pyplot as plt
from Module import Module

size = 3
steps = 1000
module = Module(size, 50)
head_out = np.zeros((steps, size))
novelty = np.zeros((steps, size))
inhibition = np.zeros((steps, size))
positive_out = np.zeros((steps, 2, size))
negative_out = np.zeros((steps, 2, size))

predictions = np.array([[1,0,0],[0,0,0]])

print("hey")

for t in range(int(steps/2)):
    module.step(predictions)
    head_out[t] = module.head_out
    positive_out[t] = module.positive_out
    negative_out[t] = module.negative_out
    novelty[t] = module.novelty

predictions = np.array([[0,0,0],[0,0,0]])

for t in range(int(steps/2), steps):
    module.step(predictions)
    head_out[t] = module.head_out
    positive_out[t] = module.positive_out
    negative_out[t] = module.negative_out
    novelty[t] = module.novelty

plt.plot(head_out[:,0])
plt.plot(positive_out[:,0,0])
plt.plot(negative_out[:,0,0])
plt.plot(novelty[:,0])
plt.show()
