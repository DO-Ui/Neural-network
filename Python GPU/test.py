import numpy as np

temp = np.ones((2, 2), dtype=np.float32)

temp3 = np.ones((3, 2), dtype=np.float32)


np.savez("test.npz", temp=temp, temp3=temp3)

temp2 = np.load("test.npz")

print(temp2["temp"])
print(temp2["temp3"])
