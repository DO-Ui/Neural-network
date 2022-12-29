#include <vector>
#include <cmath>
#include "Layer.h"
#include "DataPoint.h"
#include <iostream>

using namespace std;

class NeuralNetwork {
	vector<Layer> layers;
	private:
		
	public:
		/// @brief Initializes a new neural network with the given layer sizes
		/// @param layerSizes 
		NeuralNetwork(vector<int> layerSizes) {
			for (int i = 0; i < layerSizes.size() - 1; i++) {
				layers.push_back(Layer(layerSizes[i], layerSizes[i + 1]));
			}
		}

		/// @brief Calculates the outputs of a neural network given the inputs
		/// @param inputs 
		/// @return outputs
		double* CalculateOutputs(double inputs[], bool print = false) {
			if (print) {
				cout << "Inputs: ";
				for (int i = 0; i < layers[0].numNodesIn; i++) {
					cout << inputs[i] << " ";
				}
			}
			for (Layer layer : layers) {
				inputs = layer.CalculateOutputs(inputs);
			}
			// print outputs
			if (print) {
				cout << "Outputs: ";
				for (int i = 0; i < layers[layers.size() - 1].numNodesOut; i++) {
					cout << inputs[i] << " ";
				}
			}
			return inputs;
		}

		/// @brief Classifies the given inputs
		/// @param inputs 
		/// @return classification
		int Classify(double inputs[]) {
			double* outputs = CalculateOutputs(inputs, false);
			int maxIndex = 0;
			for (int i = 0; i < layers[layers.size() - 1].numNodesOut; i++) {
				if (outputs[i] > outputs[maxIndex]) {
					maxIndex = i;
				}
			}
			return maxIndex;
		}

		/// @brief Calculates the cost of a given data point
		/// @param dataPoint 
		/// @return cost
		double Cost(DataPoint dataPoint) {
			// convert vector to array
			double inputs[dataPoint.inputs.size()];
			for (int i = 0; i < dataPoint.inputs.size(); i++) {
				inputs[i] = dataPoint.inputs[i];
			}
			double* out_ptr = CalculateOutputs(inputs);

			double outputs[layers[layers.size() - 1].numNodesOut];
			for (int i = 0; i < layers[layers.size() - 1].numNodesOut; i++) {
				outputs[i] = out_ptr[i];
			}

			// print datapoint
			// cout << "Inputs: ";
			// for (int i = 0; i < layers[0].numNodesIn; i++) {
			// 	cout << dataPoint.inputs[i] << " ";
			// }
			// cout << endl;
			// // print expected outputs
			// cout << "Expected Outputs: ";
			// for (int i = 0; i < layers[layers.size() - 1].numNodesOut; i++) {
			// 	cout << dataPoint.expectedOutputs[i] << " ";
			// }
			// cout << endl;
			// // print outputs
			// cout << "Outputs: ";
			// for (int i = 0; i < layers[layers.size() - 1].numNodesOut; i++) {
			// 	cout << outputs[i] << " ";
			// }
			// cout << endl;

			Layer outputLayer = layers[layers.size() - 1];
			double cost = 0;
			for (int i = 0; i < outputLayer.numNodesOut; i++) {
				// cout << "output: " << outputs[i] << " expected: " << dataPoint.expectedOutputs[i] << endl;
				cost += outputLayer.NodeCost(outputs[i], dataPoint.expectedOutputs[i]);
			}
			return cost;
		}

		/// @brief Calculates the cost of a given set of data points
		/// @param dataPoints 
		/// @return 
		double Cost(vector<DataPoint> dataPoints) {
			double totalCost = 0;
			for (DataPoint dataPoint : dataPoints) {
				totalCost += Cost(dataPoint);
			}
			return totalCost / dataPoints.size();
		}

		void Learn(vector<DataPoint> trainingData, double learnRate) {
			const double h = 0.01;
			double originalCost = Cost(trainingData);
			for (Layer layer : layers) {
				for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++) {
					for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++) {
						layer.weights[nodeIn][nodeOut] += h;
						double newCost = Cost(trainingData) - originalCost;
						// cout << "newCost: " << setprecision(10) << newCost << endl;
						layer.weights[nodeIn][nodeOut] -= h;
						layer.costGradientW[nodeIn][nodeOut] = newCost / h;
					}
				}


				for (int biasIndex = 0; biasIndex < layer.bias.size(); biasIndex++) {
					layer.bias[biasIndex] += h;
					double newCost = Cost(trainingData) - originalCost;
					layer.bias[biasIndex] -= h;
					layer.costGradientB[biasIndex] = newCost / h;
				}

			}
			for (Layer layer : layers) { // Apply gradients to all layers
				layer.ApplyGradients(learnRate / trainingData.size());
			}
			// cout << "Cost: " << setprecision(10) << Cost(trainingData) << endl;


			// print bias gradients
			// cout << "Bias Gradients: " << endl;
			// for (Layer layer : layers) {
			// 	for (int biasIndex = 0; biasIndex < layer.bias.size(); biasIndex++) {
			// 		cout << layer.costGradientB[biasIndex] << " ";
			// 	}
			// 	cout << endl;
			// }


			// print layer weights
			cout << "Layer Weights: " << endl;
			for (Layer layer : layers) {
				for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++) {
					for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++) {
						cout << layer.weights[nodeIn][nodeOut] << " ";
					}
					cout << endl;
				}
			}
		}


};