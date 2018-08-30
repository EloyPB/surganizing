import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-50*(x-0.8)))


x = np.linspace(-1, 1, 100)

tanh = np.clip(np.tanh(3*x), 0, a_max=None)

sigm = sigmoid(x)

plt.plot(x, x)
plt.plot(x, tanh)
plt.plot(x, sigm)
plt.show()