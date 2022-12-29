import numpy as np

class DataPoint:
	inputs = np.array([])
	expectedOutputs = np.array([])
	label:int

	def __init__(self , inputs , label, numLabels:int):
		self.inputs = inputs
		self.label = label
		# One hot encoding, 1 for the correct label, 0 for the rest
		OneHot = np.zeros(numLabels)
		OneHot[label] = 1
		self.expectedOutputs = OneHot
