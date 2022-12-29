#include <bits/stdc++.h>

// Custom classes
#include "NeuralNetwork.h"
// #include "DataPoint.h"
using namespace std;

signed main() {


	// Generate Data:
	// uniform_real_distribution<double> dist(0, 10);  //(min, max)
	// mt19937 rng; 
	// rng.seed(random_device{}());
	// ofstream file;
	// file.open("test.txt");
	// for (int i = 0; i < 15; i++) {
	// 	double x = dist(rng);
	// 	double y = dist(rng);
	// 	if (x > 5 && y > 5) {
	// 		file << "(" << x << "," << y << ") " << 1 << endl;
	// 	} else {
	// 		file << "(" << x << "," << y << ") " << 0 << endl;
	// 	}
	// }

	// Read Data:
	ifstream in_file;
	vector<DataPoint> data;
	in_file.open("train.txt");
	string line;
	while (getline(in_file, line)) {
		stringstream ss(line);
		string temp;
		getline(ss, temp, '(');
		getline(ss, temp, ',');
		double x = stod(temp);
		getline(ss, temp, ')');
		double y = stod(temp);
		getline(ss, temp, ' ');
		getline(ss, temp);
		int label = stoi(temp);
		data.push_back(DataPoint({x, y}, label, 2));
	}
	in_file.close();
	// Train
	NeuralNetwork network = NeuralNetwork({2, 3, 2});
	for (int i = 0; i <= 10; i++) {
		if (i % 100 == 0) {
			cout << "Epoch " << i << endl;
		}
		network.Learn(data, 1000);
	}

	// Test
	vector<pair<vector<double>, int>> test_data;
	ifstream file;
	line = "";
	file.open("test.txt");
	while (getline(file, line)) {
		stringstream ss(line);
		string temp;
		getline(ss, temp, '(');
		getline(ss, temp, ',');
		double x = stod(temp);
		getline(ss, temp, ')');
		double y = stod(temp);
		getline(ss, temp, ' ');
		getline(ss, temp);
		int label = stoi(temp);
		vector<double> inputs = {x, y};
		test_data.push_back({inputs, label});
	}
	file.close();

	// print data
	// for (pair<vector<double>, int> data : test_data) {
	// 	cout << "Inputs: ";
	// 	for (int i = 0; i < 2; i++) {
	// 		cout << data.first[i] << " ";
	// 	}
	// 	cout << "Actual: " << data.second << endl;
	// }

	double accuracy = 0;
	for (pair<vector<double>, int> data : test_data) {
		// convert vector double to double array
		double* inputs = new double[data.first.size()];
		for (int i = 0; i < data.first.size(); i++) {
			inputs[i] = data.first[i];
		}
		int output = network.Classify(inputs);
		cout << "Predicted: " << output << " Actual: " << (data.second == 1 ? 0 : 1) << endl;

		accuracy += (output != data.second ? 100 : 0);

	}

	cout << "Accuracy: " << accuracy / test_data.size() << "%" << endl;
	
}