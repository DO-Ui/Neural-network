import math
from random import random
class Layer:
	numNodesIn:int
	numNodesOut:int
	costGradientW = [[]]
	costGradientB = []
	weights = [[]]
	biases = []

	activations = []
	weightedInputs = []
	inputs = []

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

		weightedInputs = [0.0 for i in range(self.numNodesOut)]

		for nodeOut in range(self.numNodesOut):
			weightedInput = self.biases[nodeOut]
			for nodeIn in range(self.numNodesIn):
				weightedInput += self.weights[nodeIn][nodeOut] * inputs[nodeIn]
			weightedInputs[nodeOut] = weightedInput

		activations = [0.0 for i in range(self.numNodesOut)]
		for nodeOut in range(self.numNodesOut):
			activations[nodeOut] = self.ActivationFunction(weightedInputs[nodeOut])

		self.inputs = inputs
		self.activations = activations
		self.weightedInputs = weightedInputs
		
		return activations
  
		# activations = [0.0 for i in range(self.numNodesOut)]
		# for nodeOut in range(self.numNodesOut):
		# 	weightedInput = self.biases[nodeOut]
		# 	for nodeIn in range(self.numNodesIn):
		# 		weightedInput += self.weights[nodeIn][nodeOut] * inputs[nodeIn]
		# 	activations[nodeOut] = self.ActivationFunction(weightedInput)
		# return activations

	def ActivationFunction(self, x:float) -> float:
		return 1.0 / (1.0 + math.exp(-x))
		# decimal.getcontext().prec = 100
		# return float(decimal.Decimal("1.0") / (decimal.Decimal("1.0") + decimal.Decimal(math.exp(-x)).exp()))

	def ActivationFunctionDerivative(self, x:float) -> float:
		activation_value = self.ActivationFunction(x)
		return activation_value * (1.0 - activation_value)


	def NodeCost(self, outputActivation:float, expectedOutput:float) -> float:
		error:float = outputActivation - expectedOutput
		return error*error

	def NodeCostDerivative(self, outputActivation:float, expectedOutput:float) -> float:
		return 2.0 * (outputActivation - expectedOutput)

	def CalculateOutputLayerNodeValues(self, expectedOutputs:list) -> list:
		nodeValues:list = [0.0 for i in range(len(expectedOutputs))]

		for nodeOut in range(len(expectedOutputs)):
			costDerivative = self.NodeCostDerivative(self.activations[nodeOut], expectedOutputs[nodeOut])
			activationDerivative = self.ActivationFunctionDerivative(self.weightedInputs[nodeOut])
			nodeValues[nodeOut] = costDerivative * activationDerivative
		return nodeValues

	def UpdateGradients(self, nodeValues:list):
		for nodeOut in range(self.numNodesOut):
			nodeValue:float = nodeValues[nodeOut]
			for nodeIn in range(self.numNodesIn):
				self.costGradientW[nodeIn][nodeOut] += nodeValue * self.inputs[nodeIn]

		for nodeOut in range(self.numNodesOut):
			self.costGradientB[nodeOut] += nodeValues[nodeOut]

		# 	self.costGradientB[nodeOut] += nodeValues[nodeOut]
		# 	for nodeIn in range(self.numNodesIn):
		# 		self.costGradientW[nodeIn][nodeOut] += nodeValues[nodeOut] * inputs[nodeIn]

	def CalculateHiddenLayerNodeValues(self, oldLayer, oldNodeValues:list) -> list:
		newNodeValues = [0.0 for i in range(self.numNodesOut)]
		for newNodeIndex in range(self.numNodesOut):
			newNodeValue = 0.0
			for oldNodeIndex in range(len(oldNodeValues)):
				weightedInputDerivative = oldLayer.weights[newNodeIndex][oldNodeIndex]
				newNodeValue += oldNodeValues[oldNodeIndex] * weightedInputDerivative
			newNodeValue *= self.ActivationFunctionDerivative(self.weightedInputs[newNodeIndex])
			newNodeValues[newNodeIndex] = newNodeValue

		return newNodeValues

	def ResetGradients(self):
		self.costGradientW = [[0.0 for i in range(self.numNodesOut)] for j in range(self.numNodesIn)]
		self.costGradientB = [0.0 for i in range(self.numNodesOut)]