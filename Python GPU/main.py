import numpy as np
import cv2
import imutils
import os
import random

from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint
import time


inputs:int = 784
outputs:int = 10
learnRate:float = 0.8

def train(load = ""):
	network = NeuralNetwork([inputs, 150, outputs])

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
			img = imutils.translate(img, random.randint(-5, 5), random.randint(-5, 5))

			# convert to 1D array
			img = img.flatten()
			# for each pixel, get value between 0.0 and 1.0
			img = img / 255.0
			trainingData.append(DataPoint(img, int(label), outputs))


	# shuffle training data
	random.shuffle(trainingData)
	# trim training data
	trainingData = trainingData[:1000]

	# count the labels in the training data
	labelCount = [0 for i in range(outputs)]
	for dataPoint in trainingData:
		labelCount[dataPoint.expectedOutputs.argmax()] += 1

	print("Distribution", labelCount)

	last_time = time.time()
	for i in range(100):
		print("Epoch", i)
		print("Time elapsed:", round(time.time() - last_time, 2), "s")
		last_time = time.time()
		if i % 15 == 0:
			network.Save(f"checkpoint_{i}")

		network.Learn(trainingData, learnRate)

	# save network
	network.Save("network")

	print("DONE!")

train()