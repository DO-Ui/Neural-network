import math
from random import random
class Layer:
	numNodesIn:int
	numNodesOut:int
	costGradientW = [[]]
	costGradientB = []
	weights = [[]]
	biases = []

	def __init__(self , numNodesIn:int , numNodesOut:int):
		self.numNodesIn = numNodesIn
		self.numNodesOut = numNodesOut
		self.costGradientW = [[0.0 for i in range(numNodesOut)] for j in range(numNodesIn)]
		self.costGradientB = [0.0 for i in range(numNodesOut)]

		self.weights = [[(random() * 2 - 1)/math.sqrt(numNodesIn) for i in range(numNodesOut)] for j in range(numNodesIn)]
		self.biases = [0.0 for i in range(numNodesOut)]

	def ApplyGradients(self, learnRate:float):
		for nodeOut in range(self.numNodesOut):
			self.biases[nodeOut] -= learnRate * self.costGradientB[nodeOut]
			for nodeIn in range(self.numNodesIn):
				self.weights[nodeIn][nodeOut] -= learnRate * self.costGradientW[nodeIn][nodeOut]

	def CalculateOutputs(self, inputs) -> list:
		activations = [0.0 for i in range(self.numNodesOut)]
		for nodeOut in range(self.numNodesOut):
			weightedInput = self.biases[nodeOut]
			for nodeIn in range(self.numNodesIn):
				weightedInput += self.weights[nodeIn][nodeOut] * inputs[nodeIn]
			activations[nodeOut] = self.ActivationFunction(weightedInput)
		return activations

	def ActivationFunction(self, x:float) -> float:
		return 1.0 / (1.0 + math.exp(-x))

	def NodeCost(self, outputActivation:float, expectedOutput:float) -> float:
		error:float = outputActivation - expectedOutput
		return error*error

