#include <vector>
#include <cmath>
#include <random>

using namespace std;

class Layer {
	private:
		double ActivationFunction(double weightedInput) {
			return 1 / (1 + exp(-weightedInput));
		}

		void InitRandomWeights() {
			for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
				for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
					uniform_real_distribution<double> dist(-1, 1);  //(min, max)
					mt19937 rng; 
					rng.seed(random_device{}());
					weights[nodeIn][nodeOut] = dist(rng);
				}
			}
		}

	public:
		int numNodesIn, numNodesOut;
		vector<vector<double>> costGradientW;
		vector<double> costGradientB;

		vector<vector<double>> weights;
		vector<double> bias;

		/// @brief Initializes a new layer with the given number of nodes in and out
		/// @param numNodesIn 
		/// @param numNodesOut 
		Layer(int numNodesIn, int numNodesOut) {
			costGradientW.resize(numNodesIn);
			for (int i = 0; i < numNodesIn; i++) {
				costGradientW[i].resize(numNodesOut);
			}
			weights.resize(numNodesIn);
			for (int i = 0; i < numNodesIn; i++) {
				weights[i].resize(numNodesOut);
			}
			costGradientB.resize(numNodesOut);
			bias.resize(numNodesOut);
			this->numNodesOut = numNodesOut;
			this->numNodesIn = numNodesIn;
			InitRandomWeights();
		}

		/// @brief Calculates the outputs of the layer given the inputs
		/// @param inputs
		/// @return outputs
		double* CalculateOutputs(double inputs[]) {
			double* outputs = new double[numNodesOut];
			for (int i = 0; i < numNodesOut; i++) {
				double weightedInput = bias[i];
				for (int j = 0; j < numNodesIn; j++) {
					weightedInput += inputs[j] * weights[j][i];
				}
				outputs[i] = ActivationFunction(weightedInput);
			}
			return outputs;
		}

		/// @brief Calculates the error of a node given the output and target
		/// @param output 
		/// @param target 
		/// @return error (squared)
		double NodeCost(double output, double target) {
			double error = target - output;
			return error * error;
		}

		/// @brief Updates weights and biases based on the cost gradient
		/// @param learnRate 
		void ApplyGradients(double learnRate) {

			// for (int i = 0; i < weights.size(); i++) {
			// 	for (int j = 0; j < weights[i].size(); j++) {
			// 		weights[i][j] -= costGradientW[i][j] * learnRate;
			// 	}
			// }


			for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
			{
				bias[nodeOut] -= costGradientB[nodeOut] * learnRate;
				for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) 
				{
					weights[nodeIn][nodeOut] -= costGradientW[nodeIn][nodeOut] * learnRate;
				}
			}
		}

		void UpdateGradients(vector<double> nodeValues) {
			for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
				for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
					double derivativeCostWrtWeight = nodeValues[nodeIn] * nodeValues[nodeOut];
					costGradientW[nodeIn][nodeOut] += derivativeCostWrtWeight;
				}
			}
		}



};