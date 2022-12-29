import numpy as np
import cv2
import imutils
import os
import random

from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint
import time
import matplotlib.pyplot as plt

inputs:int = 784
outputs:int = 10
learnRate:float = 1

network_structure = [inputs, outputs]

batch_size = 100

def accuracy(network:NeuralNetwork, data):

	data_in = []
	for i in range(len(data)):
		data_in.append(data[i][0])

	results = []

	for i in range(len(data_in)):
		results.append(network.Classify(data_in[i]))


	accuracy:float = 0.0
	tests:int = 0

	for i in range(len(results)):
		if results[i] == data[i][1]:
			accuracy += 100
		tests += 1

	accuracy /= tests

	return accuracy



def train(load = ""):
	network = NeuralNetwork(network_structure)

	if load != "":
		network.Load(load)

	trainingData = []

	trainData = os.walk("/home/doui/Documents/Code-Stuff/Ai/digits/train")

	for root, dirs, files in trainData:
		# print(dirs)
		for file in files:
			label = root.split("/")[-1]
			# print(label)
			img = cv2.imread(root + "/" + file, cv2.IMREAD_GRAYSCALE)
			img = imutils.rotate(img, random.randint(-15, 15))
			img = imutils.translate(img, random.randint(-5, 5), random.randint(-3, 3))
			# randomly scale image
			scale = random.uniform(0.8, 1.2)
			img = imutils.resize(img, width=int(img.shape[1] * scale), height=int(img.shape[0] * scale))

			# refit image to 28x28
			img = cv2.resize(img, (28, 28))

			# add noise to image
			for i in range(0, random.randint(15, 100)):
				x = random.randint(0, 27)
				y = random.randint(0, 27)
				img[x][y] = random.randint(0, 255)

			# cv2.imshow("img", img)
			# cv2.waitKey(0)

			# convert to 1D array
			img = img.flatten()
			# for each pixel, get value between 0.0 and 1.0
			img = img / 255.0
			trainingData.append(DataPoint(img, int(label), outputs))

	### Calculate Accuracy ###
	testData = os.walk("/home/doui/Documents/Code-Stuff/Ai/digits/test")


	data = []

	for root, dirs, files in testData:
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
			data.append([img, int(label)])

	random.shuffle(data)

	data_in = []
	for i in range(len(data)):
		data_in.append(data[i][0])

	

	# shuffle training data
	random.shuffle(trainingData)

	# split into batches
	batches = []
	for i in range(0, len(trainingData), batch_size):
		batches.append(trainingData[i:i+batch_size])

	print("Epochs:", len(batches))

	last_time = time.time()
	# for i in range():

	plt.ion()
	# interactive(True)
 
	plt.show()

	accuracy_list = []

	for i, batch in enumerate(batches):
		print("Epoch", i)
		print("Time elapsed:", round(time.time() - last_time, 2), "s")
		last_time = time.time()
		if i % 50 == 0:
			network.Save(f"checkpoint_{i}")
		if i % 10 == 0:
			acc = accuracy(network, data)
			print("Accuracy:", round(acc, 2), "%")
			accuracy_list.append(acc)
			accuracy_x = np.arange(0, len(accuracy_list))


			plt.plot(accuracy_x, accuracy_list, color="red", marker='x')
			plt.draw()

			cost = network.TotalCost(batch)
			print("Total cost:", cost)

			# cost_list.append(cost)
			# cost_x = np.arange(0, len(cost_list))

			# plt.plot(cost_x, cost_list, color="green", marker='x')
			# plt.draw()

			# plt.draw()
			plt.legend(["Accuracy"])
			plt.pause(0.01)

			# plt.show()
			# input("Press enter to continue...")

		network.Learn(batch, learnRate)
		# acc = accuracy(network, data)
		# print("Accuracy:", round(acc, 2), "%")

	# save network
	network.Save("network")

	print("DONE!")



def test(load):
	network = NeuralNetwork(network_structure)
	network.Load(load)

	testData = os.walk("/home/doui/Documents/Code-Stuff/Ai/digits/test")


	data = []

	for root, dirs, files in testData:
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
			data.append([img, int(label)])


	data_in = []
	for i in range(len(data)):
		data_in.append(data[i][0])

	results = []

	for i in range(len(data_in)):
		results.append(network.Classify(data_in[i]))

	# with ThreadPoolExecutor(max_workers=4) as executor:
	# 	results = executor.map(network.Classify, data_in)
  
	accuracy:float = 0.0
	tests:int = 0

	for i in range(len(results)):
		print("Expected:", data[i][1], "Got:", results[i])
		if results[i] == data[i][1]:
			accuracy += 100
		tests += 1
	

			# result = network.Classify(img)
			# print("Result: " + str(result), end=" ")
			# print("Expected: " + str(int(label)))
			# if result == int(label):
			# 	accuracy += 100
			# tests += 1

	accuracy /= tests
	print("Accuracy: " + str(round(accuracy, 2)) + "%")

train()
# train("checkpoint_100.npz")
# test("network.npz")
# test("checkpoint_50.npz")