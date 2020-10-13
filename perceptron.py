import numpy as np
from matplotlib import pyplot as plt

A = np.array([1, 2, 3, 4, 35, 23, 123.1], dtype='float32')

A = np.random.randn(100, 2) + [3, 1]
B = np.random.randn(100, 2) + [-2, 0.5]
E = np.zeros((200, 2))
E[0:100] = A
E[100:200] = B
t = np.zeros((200,))
t[0:100] = -1
t[100:200] = 1

x = A[:, 0]
y = A[:, 1]
plt.scatter(x, y)
plt.scatter(B[:,0], B[:,1])
plt.plot([4, -4], [3, 5])

plt.scatter(E[:,0], E[:,1], c=t)

plt.show()
