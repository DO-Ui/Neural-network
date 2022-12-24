from Layer import Layer
class NeuralNetwork:
	layers:Layer = []

	def __init__(self, layerSizes:list):
		self.layers = [Layer for i in range(len(layerSizes)-1)]
		for i in range(len(self.layers)):
			self.layers[i] = Layer(layerSizes[i], layerSizes[i+1])


	def CalculateOutputs(self, inputs:list) -> list:
		for layer in self.layers:
			inputs = layer.CalculateOutputs(inputs)
		return inputs

	def Classify(self, inputs:list) -> int:
		outputs = self.CalculateOutputs(inputs)
		return outputs.index(max(outputs))

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

	def Learn(self, trainingData:list, learnRate:float):
		h:float = 0.0001
		originalCost:float = self.TotalCost(trainingData)

		for layer in self.layers:
			for nodeIn in range(layer.numNodesIn):
				for nodeOut in range(layer.numNodesOut):
					layer.weights[nodeIn][nodeOut] += h
					deltaCost:float = self.TotalCost(trainingData) - originalCost
					layer.weights[nodeIn][nodeOut] -= h
					layer.costGradientW[nodeIn][nodeOut] = deltaCost / h

			for biasIndex in range(len(layer.biases)):
				layer.biases[biasIndex] += h
				deltaCost:float = self.TotalCost(trainingData) - originalCost
				layer.biases[biasIndex] -= h
				layer.costGradientB[biasIndex] = deltaCost / h

		for layer in self.layers:
			layer.ApplyGradients(learnRate)

		# print layers
		# for idx, layer in enumerate(self.layers):
		# 	print("Weights for layer", idx, ":")
		# 	for nodeIn in range(layer.numNodesIn):
		# 		print(layer.weights[nodeIn])



