from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint

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
for i in range(1000):
	if i % 10 == 0:
		print("Epoch", i)
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