from Layer import Layer
from DataPoint import DataPoint
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class NeuralNetwork:
	layers = []

	def __init__(self, layerSizes:list):
		self.layers = [Layer for i in range(len(layerSizes)-1)]
		for i in range(len(self.layers)):
			self.layers[i] = Layer(layerSizes[i], layerSizes[i+1])

	# multiprocessing could be used here
	def CalculateOutputs(self, inputs:list) -> list:
		for layer in self.layers:
			inputs = layer.CalculateOutputs(inputs)
		return inputs

	def Classify(self, inputs:list) -> int:
		outputs = self.CalculateOutputs(inputs)
		return outputs.index(max(outputs))


	##### not really needed, only really used for debugging

	def Cost(self, dataPoint) -> float:
		outputs = self.CalculateOutputs(dataPoint.inputs)
		cost:float = 0.0
		for nodeOut in range(len(outputs)):
			cost += self.layers[-1].NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut])
		return cost

	def TotalCost(self, dataPoints:list) -> float:
		totalCost:float = 0.0
		
		for dataPoint in dataPoints:
			totalCost += self.Cost(dataPoint)


		return totalCost / len(dataPoints)

	#####

	# def Learn(self, trainingData:list, learnRate:float):
	# 	h:float = 0.0001
	# 	originalCost:float = self.TotalCost(trainingData)

	# 	for layer in self.layers:
	# 		for nodeIn in range(layer.numNodesIn):
	# 			for nodeOut in range(layer.numNodesOut):
	# 				layer.weights[nodeIn][nodeOut] += h
	# 				deltaCost:float = self.TotalCost(trainingData) - originalCost
	# 				layer.weights[nodeIn][nodeOut] -= h
	# 				layer.costGradientW[nodeIn][nodeOut] = deltaCost / h

	# 		for biasIndex in range(len(layer.biases)):
	# 			layer.biases[biasIndex] += h
	# 			deltaCost:float = self.TotalCost(trainingData) - originalCost
	# 			layer.biases[biasIndex] -= h
	# 			layer.costGradientB[biasIndex] = deltaCost / h

	# 	for layer in self.layers:
	# 		layer.ApplyGradients(learnRate)

	def Learn(self, trainingData:list, learnRate:float):
		for dataPoint in trainingData:
			self.UpdateAllGradients(dataPoint)

		# apply gradients multiprocessing
		# for layer in self.layers:
		# 	p = multiprocessing.Process(target=layer.ApplyGradients, args=(learnRate / len(trainingData),))
		# 	p.start()
		# 	p.join()

		# with ThreadPoolExecutor() as executor:
		# 	executor.map(layer.applyGradients, [learnRate / len(trainingData) for layer in self.layers])
			# for layer in self.layers:
			# 	executor.submit(layer.ApplyGradients, learnRate / len(trainingData))

		for layer in self.layers:
			layer.ApplyGradients(learnRate / len(trainingData))

		# clear gradients
		for layer in self.layers:
			layer.ResetGradients()

		# print avg cost
		# print(self.TotalCost(trainingData))



	def UpdateAllGradients(self, dataPoint:DataPoint):

		inputsToNextLayer = dataPoint.inputs
		# multiprocess this
		for layer in self.layers:
			inputsToNextLayer = layer.CalculateOutputs(inputsToNextLayer)
		

		# self.CalculateOutputs(dataPoint.inputs)

		outputLayer = self.layers[-1]
		nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs)
		outputLayer.UpdateGradients(nodeValues) # note this too

		for hiddenLayerIndex in range(len(self.layers)-2, -1, -1):
			hiddenLayer = self.layers[hiddenLayerIndex]
			nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(self.layers[hiddenLayerIndex+1], nodeValues)
			hiddenLayer.UpdateGradients(nodeValues) # double check this

	def Save(self, filename:str):
		file = open(filename, "w")
		for layer in self.layers:
			file.write(str(layer.weights) + "\n")
			file.write(str(layer.biases) + "\n")
		file.close()

	def Load(self, filename:str):
		file = open(filename, "r")
		for layer in self.layers:
			layer.weights = np.array(eval(file.readline()))
			layer.biases = np.array(eval(file.readline()))
		file.close()