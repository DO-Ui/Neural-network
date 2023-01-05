import math
from numba import jit

@jit
def ActivationFunction(weightedInput:float) -> float:
	return 1.0 / (1.0 + math.exp(-weightedInput)) # sigmoid
	# return weightedInput if weightedInput > 0 else 0 # ReLU

@jit
def ActivationFunctionDerivative(weightedInput:float) -> float:
	# sigmoid derivative
	activation:float = ActivationFunction(weightedInput)
	return activation * (1.0 - activation)
	# return 1 if weightedInput > 0 else 0 # ReLU derivative

@jit
def NodeCostDerivative(outputActivation:float, expectedOutput:float) -> float:
	return 2 * (outputActivation - expectedOutput)