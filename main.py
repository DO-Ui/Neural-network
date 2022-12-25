from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint

import time
import matplotlib.pyplot as plt
import cv2
import imutils
import os
import random

inputs:int = 784
outputs:int = 10
learnRate:float = 0.1

trainingData = []

trainData = os.walk("/home/doui/Documents/Code-Stuff/Ai/digits/train")

for root, dirs, files in trainData:
	# print(dirs)
	for file in files:
		label = root.split("/")[-1]
		# print(label)
		img = cv2.imread(root + "/" + file, cv2.IMREAD_GRAYSCALE)
		img = imutils.rotate(img, random.randint(-15, 15))
		img = imutils.translate(img, random.randint(-3, 3), random.randint(-3, 3))

		# convert to 1D array
		img = img.flatten()
		# for each pixel, get value between 0.0 and 1.0
		img = img / 255.0
		trainingData.append(DataPoint(img, int(label), outputs))


# shuffle training data
trainingData = random.sample(trainingData, len(trainingData))
# trim training data
trainingData = trainingData[:500]

network = NeuralNetwork([inputs, 150, outputs])
last_time = time.time()
for i in range(20):
	if i % 1 == 0:
		print("Epoch", i)
		print("Time elapsed:", round(time.time() - last_time, 2), "s")
		last_time = time.time()
		# cost = network.TotalCost(trainingData)
		# print("Cost:", cost)
		# plt.plot(i, cost, "ro")
	network.Learn(trainingData, learnRate)

# save network
network.Save("network.txt")


# # load network
# network.Load("network.txt")

# testData = os.walk("/home/doui/Documents/Code-Stuff/Ai/digits/test")

# accuracy:float = 0.0

# for root, dirs, files in testData:
# 	for file in files:
# 		label = root.split("/")[-1]
# 		# print(label)
# 		img = cv2.imread(root + "/" + file, cv2.IMREAD_GRAYSCALE)
# 		img = imutils.rotate(img, random.randint(-15, 15))
# 		img = imutils.translate(img, random.randint(-3, 3), random.randint(-3, 3))

# 		# convert to 1D array
# 		img = img.flatten()
# 		# for each pixel, get value between 0.0 and 1.0
# 		img = img / 255.0
# 		result = network.Classify(img)
# 		print("Result: " + str(result), end=" ")
# 		print("Expected: " + str(int(label)))
# 		if result == int(label):
# 			accuracy += 100

# accuracy /= len(trainingData)
# print("Accuracy: " + str(round(accuracy, 2)) + "%")






# img = cv2.imread("/home/doui/Documents/Code-Stuff/Ai/digits/test/5/img_11.jpg", cv2.IMREAD_GRAYSCALE)
# img = img.flatten()
# img = img / 255.0
# print(network.Classify(img))


# plt.show()