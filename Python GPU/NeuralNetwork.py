import numpy as np
from numba import cuda, jit
from Layer import Layer
from DataPoint import DataPoint

class NeuralNetwork:
	layers = []

	def __init__(self, layerSizes:list):
		self.layers = [Layer for i in range(len(layerSizes)-1)]
		for i in range(len(self.layers)):
			self.layers[i] = Layer(layerSizes[i], layerSizes[i+1])

	##### LEARNING #####

	def Learn(self, trainingData:list, learnRate:float):
		for dataPoint in trainingData:
			self.UpdateAllGradients(dataPoint)

		for layer in self.layers:
			### Copy over memory to GPU ###
			device_costGradientW = cuda.to_device(layer.costGradientW)
			device_costGradientB = cuda.to_device(layer.costGradientB)
			device_weights = cuda.to_device(layer.weights)
			device_biases = cuda.to_device(layer.biases)

			### Apply Gradients ###
			layer.ApplyGradients[layer.numNodesOut, 1](device_costGradientW, device_costGradientB, device_weights, device_biases, learnRate/len(trainingData), layer.numNodesIn) # Execute 1 thread per node in parallel

			### Copy back to CPU ###
			layer.weights = device_weights.copy_to_host()
			layer.biases = device_biases.copy_to_host()

		for layer in self.layers:
			layer.ResetGradients()


	def UpdateAllGradients(self, dataPoint:DataPoint):
		inputsToNextLayer = dataPoint.inputs
		### Calculate outputs ###
		for layer in self.layers:
			### Copy over memory to GPU ###
			device_activations = cuda.to_device(np.zeros(layer.numNodesOut, dtype=np.float32))
			device_weightedInputs = cuda.to_device(np.zeros(layer.numNodesOut, dtype=np.float64))

			device_biases = cuda.to_device(layer.biases)
			device_weights = cuda.to_device(layer.weights)

			layer.CalculateOutputs[layer.numNodesOut, 1](inputsToNextLayer, device_weightedInputs, device_activations, device_biases, device_weights, layer.numNodesIn) # Execute 1 thread per node in parallel
			layer.inputs = inputsToNextLayer

			### Copy back to CPU ###
			activations = device_activations.copy_to_host()
			layer.weightedInputs = device_weightedInputs.copy_to_host()
			layer.activations = activations


			### Update inputs for next layer ###
			inputsToNextLayer = activations

		### Calculate gradients ###

		outputLayer = self.layers[-1]
		nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs)

		### Copy over memory to GPU ###
		device_inputs = cuda.to_device(outputLayer.inputs)
		device_nodeValues = cuda.to_device(nodeValues)
		device_costGradientW = cuda.to_device(outputLayer.costGradientW)
		device_costGradientB = cuda.to_device(outputLayer.costGradientB)

		### Process on GPU ###
		outputLayer.UpdateGradients[outputLayer.numNodesOut, 1](device_inputs, device_nodeValues, device_costGradientW, device_costGradientB, outputLayer.numNodesIn) # Execute 1 thread per node in parallel

		### Copy back to CPU ###
		outputLayer.costGradientW = device_costGradientW.copy_to_host()
		outputLayer.costGradientB = device_costGradientB.copy_to_host()

		### Calculate gradients for hidden layers ###
		for i in range(len(self.layers)-2, -1, -1):
			hiddenLayer = self.layers[i]
			nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(self.layers[i+1], nodeValues)

			### Copy over memory to GPU ###
			device_inputs = cuda.to_device(hiddenLayer.inputs)
			device_nodeValues = cuda.to_device(nodeValues)
			device_costGradientW = cuda.to_device(hiddenLayer.costGradientW)
			device_costGradientB = cuda.to_device(hiddenLayer.costGradientB)

			### Process on GPU ###
			hiddenLayer.UpdateGradients[hiddenLayer.numNodesOut, 1](device_inputs, device_nodeValues, device_costGradientW, device_costGradientB, hiddenLayer.numNodesIn) # Execute 1 thread per node in parallel

			### Copy back to CPU ###
			hiddenLayer.costGradientW = device_costGradientW.copy_to_host()
			hiddenLayer.costGradientB = device_costGradientB.copy_to_host()


	##### CLASSIFYING #####

	def CalculateOutputs(self, inputs:np.array) -> np.array:
		for layer in self.layers:
			### Copy over memory to GPU ###
			device_activations = cuda.to_device(np.zeros(layer.numNodesOut, dtype=np.float32))
			device_weightedInputs = cuda.to_device(np.zeros(layer.numNodesOut, dtype=np.float64))

			device_biases = cuda.to_device(layer.biases)
			device_weights = cuda.to_device(layer.weights)

			### Process on GPU ###
			layer.CalculateOutputs[layer.numNodesOut, 1](inputs, device_weightedInputs, device_activations, device_biases, device_weights, layer.numNodesIn) # Execute 1 thread per node in parallel

			### Copy back to CPU ###

			activations = device_activations.copy_to_host()
			layer.weightedInputs = device_weightedInputs.copy_to_host()
			layer.activations = activations
			layer.inputs = inputs

			### Update inputs for next layer ###
			inputs = activations

		return inputs


	def Classify(self, inputs:np.array) -> int:
		outputs = self.CalculateOutputs(inputs)
		return outputs.argmax()


	def Save(self, fileName:str):
		weights = np.array([layer.weights for layer in self.layers], dtype=object)
		biases = np.array([layer.biases for layer in self.layers], dtype=object)
		np.savez(fileName, weights=weights, biases=biases)

	def Load(self, fileName:str):
		data = np.load(fileName, allow_pickle=True)
		for i in range(len(self.layers)):
			self.layers[i].weights = data['weights'][i]
			self.layers[i].biases = data['biases'][i]


	##### DEBUGGING #####

	def Cost(self, dataPoint:DataPoint) -> float:
		outputs = self.CalculateOutputs(dataPoint.inputs)
		cost:float = 0.0
		# nodecosts = np.zeros(len(outputs))

		# ### Copy over memory to GPU ###
		# device_outputs = cuda.to_device(outputs)
		# device_expectedOutputs = cuda.to_device(dataPoint.expectedOutputs)
		# device_nodecosts = cuda.to_device(nodecosts)

		# ### Process on GPU ###
		# self.layers[-1].NodeCost[len(outputs), 1](device_outputs, device_expectedOutputs, device_nodecosts) # Execute 1 thread per node in parallel

		# ### Copy back to CPU ###
		# nodecosts = device_nodecosts.copy_to_host()

		# for i in range(len(outputs)):
		# 	cost += nodecosts[i]
  
		for i in range(len(outputs)):
			cost += self.layers[-1].NodeCost(outputs[i], dataPoint.expectedOutputs[i])

		return cost

	def TotalCost(self, dataPoints) -> float:
		totalCost:float = 0.0
		for dataPoint in dataPoints:
			totalCost += self.Cost(dataPoint)

		return totalCost / len(dataPoints)
