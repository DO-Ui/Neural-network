import numpy as np

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# import matplotlib.pyplot as plt
data = spiral_data(5000, 3)

with open("/home/doui/Documents/Code-Stuff/Ai/train.txt", "w") as file:
	for i in range(len(data[0])):
		file.write(f"({data[0][i][0]},{data[0][i][1]}) {data[1][i]}\n")
