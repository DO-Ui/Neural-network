import numpy as np
import math
from numba import cuda, jit

class Layer:
	numNodesIn:int
	numNodesOut:int
	costGradientW:np.array
	costGradientB:np.array
	weights:np.array
	biases:np.array

	activations:np.array
	weightedInputs:np.array
	inputs:np.array

	def __init__(self , numNodesIn:int , numNodesOut:int):
		self.numNodesIn = numNodesIn
		self.numNodesOut = numNodesOut
		self.costGradientW = np.zeros((numNodesIn, numNodesOut), dtype=np.float32)
		self.costGradientB = np.zeros(numNodesOut, dtype=np.float32)

		self.weights = np.random.rand(numNodesIn, numNodesOut).astype(np.float32)
		self.weights = (self.weights * 2 - 1) / np.sqrt(numNodesIn)
		self.biases = np.zeros(numNodesOut, dtype=np.float32)

	##### HELPERS #####

	# @jit
	# def ActivationFunction(self, weightedInput:float) -> float:
	# 	return 1 / (1 + math.exp(-weightedInput))

	# @jit
	# def ActivationFunctionDerivative(self, weightedInput:float) -> float:
	# 	return np.exp(-weightedInput) / np.square(1 + np.exp(-weightedInput))

	# @jit
	# def NodeCost(self, outputActivation:float, expectedOutput:float) -> float:
	# 	error:float = outputActivation - expectedOutput
	# 	return error * error

	# @jit
	# def NodeCostDerivative(self, outputActivation:float, expectedOutput:float) -> float:
	# 	return 2 * (outputActivation - expectedOutput)

	@staticmethod
	@cuda.jit
	def CalculateOutputLayerNodeValuesHelper(activations:np.array, weightedInputs:np.array, expectedOutputs:np.array, nodeValues:np.array):
		nodeOut = cuda.grid(1)
		activation_value = 1.0 / (1.0 + math.exp(-weightedInputs[nodeOut]))
		nodeValues[nodeOut] = (2*(activations)[nodeOut] - (expectedOutputs)[nodeOut]) * (activation_value) * (1.0 - activation_value)

	##### OUTPUTS #####

	@staticmethod
	@cuda.jit
	def CalculateOutputs(inputs:np.array, weightedInputs:np.array, activations:np.array, biases:np.array, weights:np.array, numNodesIn:int):
		
		nodeOut = cuda.grid(1)
		weightedInput = biases[nodeOut]
		for nodeIn in range(numNodesIn):
			weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut]
		weightedInputs[nodeOut] = weightedInput

		activations[nodeOut] = 1 / (1 + math.exp(-weightedInput))

	def CalculateOutputLayerNodeValues(self, expectedOutputs:np.array) -> np.array:
		nodeValues = np.zeros(len(expectedOutputs), dtype=np.float32)

		### Copy to GPU ###
		device_activations = cuda.to_device(self.activations)
		device_weightedInputs = cuda.to_device(self.weightedInputs)
		device_expectedOutputs = cuda.to_device(expectedOutputs)
		device_nodeValues = cuda.to_device(nodeValues)

		### Calculate ###
		self.CalculateOutputLayerNodeValuesHelper[len(expectedOutputs), 1](device_activations, device_weightedInputs, device_expectedOutputs, device_nodeValues)

		### Copy to CPU ###
		nodeValues = device_nodeValues.copy_to_host()

		return nodeValues

	def CalculateHiddenLayerNodeValues(self, oldLayer, oldNodeValues) -> np.array:
		newNodeValues = np.zeros(self.numNodesOut, dtype=np.float32)
		for newNodeIndex in range(self.numNodesOut):
			newNodeValue = 0.0
			for oldNodeIndex in range(len(oldNodeValues)):
				weightedInputDerivative = oldLayer.weights[newNodeIndex, oldNodeIndex]
				newNodeValue += oldNodeValues[oldNodeIndex] * weightedInputDerivative

			# newNodeValue *= self.ActivationFunctionDerivative(self.weightedInputs[newNodeIndex])
			# newNodeValues[newNodeIndex] = newNodeValue

			weightedInput = self.weightedInputs[newNodeIndex]

			# if weightedInput > 1:
			# 	print("panic")

			activation = 1.0 / (1.0 + math.exp(-weightedInput))
			newNodeValues[newNodeIndex] = newNodeValue * activation * (1.0 - activation)

		return newNodeValues

	##### GRADIENTS #####

	@staticmethod
	@cuda.jit
	def UpdateGradients(inputs:np.array, nodeValues:np.array, costGradientW:np.array, costGradientB:np.array, numNodesIn:int):
		nodeOut = cuda.grid(1)
		for nodeIn in range(numNodesIn):
			costGradientW[nodeIn,nodeOut] += inputs[nodeIn] * nodeValues[nodeOut]
		costGradientB[nodeOut] += nodeValues[nodeOut]

	@staticmethod
	@cuda.jit
	def ApplyGradients(costGradientW:np.array, costGradientB:np.array, weights:np.array, biases:np.array, learningRate:float, numNodesIn:int):
		nodeOut = cuda.grid(1)
		biases[nodeOut] -= learningRate * costGradientB[nodeOut]
		for nodeIn in range(numNodesIn):
			weights[nodeIn,nodeOut] -= learningRate * costGradientW[nodeIn,nodeOut]

	def ResetGradients(self):
		self.costGradientW = np.zeros((self.numNodesIn, self.numNodesOut), dtype=np.float32)
		self.costGradientB = np.zeros(self.numNodesOut, dtype=np.float32)
		self.weightedInputs = np.zeros(self.numNodesOut, dtype=np.float32)

