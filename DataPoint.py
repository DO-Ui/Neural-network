class DataPoint:
	inputs = []
	expectedOutputs = []
	label:int

	def __init__(self , inputs , label, numLabels:int):
		self.inputs = inputs
		self.label = label
		self.expectedOutputs = [1 if i == label else 0 for i in range(numLabels)]

