from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint
import time

trainingData = []

with open("train.txt", "r") as file:
	data = file.readlines()
	for line in data:
		label = int(line.split()[-1])
		_point = line.split()[0].strip("()").split(",")
		x = float(_point[0])
		y = float(_point[1])
		trainingData.append(DataPoint([x, y], label, 2))


network = NeuralNetwork([2, 3, 2])
last_time = time.time()
for i in range(500):
	if i % 10 == 0:
		print("Epoch", i)
		print("Time elapsed:", round(time.time() - last_time, 2), "s")
		last_time = time.time()
	network.Learn(trainingData, 0.8)


with open("test.txt", "r") as file:
	data = file.readlines()

	accuracy:float = 0.0
	for line in data:
		_point = line.split()[0].strip("()").split(",")
		x = float(_point[0])
		y = float(_point[1])
		result = network.Classify([x, y])
		print("Result: " + str(result), end=" ")
		print("Expected: " + str(int(line.split()[-1])))
		if result == int(line.split()[-1]):
			accuracy += 100
	accuracy /= len(data)
	print("Accuracy: " + str(round(accuracy, 2)) + "%")