import numpy as np

m = np.random.rand(5, 1)

x = np.random.rand(500,1)

print(np.sum(np.dot(x, m.T)))