#include <vector>
#include <cmath>
using namespace std;
class DataPoint {
	public:
		vector<double> inputs;
		vector<double> expectedOutputs;
		int label;

		/// @brief inputs is the data, label is the expected output
		/// @param inputs 
		/// @param label 
		/// @param numLabels 
		DataPoint(vector<double> inputs, int label, int numLabels) { // input is the data, label is the expected output
			this->inputs = inputs;
			this->label = label;
			this->expectedOutputs = CreateOneHot(label, numLabels);
		}

		vector<double> CreateOneHot(int index, int num) {
			vector<double> oneHot(num);
			oneHot[index] = 1;
			return oneHot;
		}

};