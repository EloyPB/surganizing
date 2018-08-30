import numpy as np
import matplotlib.pyplot as plt

h = np.array([1, 0])
w = np.zeros(2)
w_log = []
s = -1
s_log = []
noise = np.random.uniform(0.7, 1.3, 2)

for t in range(1000):
    h_noise = h*noise
    if t % 50 == 0:
        noise = np.random.uniform(0.7, 1.3, 2)
    s += (-s - 1 + np.dot(h_noise, w))/10
    s_log.append(s + 0)
    w = w - 0.01*h_noise*s
    w_log.append(w)

h = np.array([1, 1])
for t in range(20000):
    h_noise = h*noise
    if t % 50 == 0:
        noise = np.random.uniform(0.7, 1.3, 2)
    s += (-s - 1 + np.dot(h_noise, w))/10
    s_log.append(s + 0)
    w = w - 0.01*h_noise*s
    w_log.append(w)


for i in range(2):
    plt.plot(np.array(w_log)[:, i])

plt.figure()
plt.plot(s_log)

plt.show()
