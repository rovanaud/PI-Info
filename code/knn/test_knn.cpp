#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
// #include "dataset/dataset.hpp"
#include "KnnClassification.hpp"
#include "../confusion_matrix/confusion_matrix.hpp"  // not here
#include "test_knn.hpp"
#include "../utils.hpp"

#include <cstdlib>
#include <cassert>
using namespace std;

/** @file
 * Test suite for the KNN class.
 * This executable will put the two provided CSV files (train and test) in objects of class Dataset, perform kNN classification on the provided column with the user-provided k, and print the resulting test set ConfusionMatrix.
*/

// Prints correct usage
void msgleaveknn(char* argv[]) {
        std::cout << "Usage: " << argv[0] << " <train_file> <test_file>  <val> [<action>]" << '\n';;
        std::cout << "action: 0 for fixed k, 1 for fine_tune" << '\n';;
}

int test_knn(int argc, char* argv[]) {
    // Tests correct usage
	if (argc < 4) {
		msgleaveknn(argv);
		return 1;
	}
	
    
    int *K;
    int val = atoi(argv[3]);
    cout << "val = " << val << endl;
    bool fine_tune = (argc == 5) ? atoi(argv[4]) : 0;

    if (fine_tune){
        K = new int[val];
        RandomNumberGenerator generator;
        for (int i = 1; i < val; i++)  K[i] = generator.generateRandomInteger(1, 2*val);
    } else {
        K = new int[1];
        K[0] = val;
    }
	
    // Puts train and test files in a Dataset object
    cout << "Reading Datasets         [/ ... " << '\n';
	Dataset train_dataset(argv[1]);
	Dataset class_dataset(argv[2]);
    cout << "Finished Datasets Reading /]" << '\n';
    
    // Prints dimension
	train_dataset.Show(false);  // only dim and samples

    // Checks if train and test are same format
	assert(train_dataset.GetDim() == class_dataset.GetDim());
	
    ofstream out;
    string filename = generateFileName(argv[0]);
    // Classification

    ConfusionMatrix best_cm;
    int best_k = -1;
    
    for (int k = 0; k < val; k++){
        ConfusionMatrix confusion_matrix;
        
        std::cout<< "Computing k-NN classification for (k="<< K[k] << ")..."<<'\n';;
        
        clock_t t = clock();
        KnnClassification knn_class(K[k], &train_dataset, 0);
        t = clock() - t;
        out.open(filename, std::ios::app);
        out << "Knn Training Execution time : " << (t*1000)/CLOCKS_PER_SEC << "ms" << '\n';
        out.close();
        // ConfusionMatrix
    
        // Starts predicting
        std::cout<< "Prediction and Confusion Matrix filling" <<'\n';;
        t = clock();
        for (int i=0; i<class_dataset.GetNbrSamples(); i++) {
            std::vector<double> sample = class_dataset.GetInstance(i);
            ANNpoint query = annAllocPt(class_dataset.GetDim()-1, 0);
            int true_label = -1;  // To not leave it uninitialized + will error in AddPrediction
            for (int j=0, j2=0; j<train_dataset.GetDim()-1 && j2<train_dataset.GetDim(); j++, j2++) {
                if (j==0 && j2==0) {
                    true_label = sample[j2];
                    j--;
                    continue;
                }
                query[j] = sample[j2];
            }
            int predicted_label = knn_class.Estimate(query, 0.25);
            annDeallocPt(query);
            confusion_matrix.AddPrediction(true_label, predicted_label);
        }
        
        t = clock() - t;

        cout <<endl
            <<"execution time: "
            <<(t*1000)/CLOCKS_PER_SEC
            <<"ms\n\n";

        out.open(filename, std::ios::app);

        out << "Computing k-NN classification for (k="<< K[k] << ")..."<<'\n';;
        out << "Knn Prediction Execution time : " << (t*1000)/CLOCKS_PER_SEC << "ms" << '\n';
        out << "Confusion Matrix : " << '\n';
        out.close();

        confusion_matrix.PrintEvaluation(filename);
        confusion_matrix.PrintEvaluation();

        if (isBetter(best_cm, confusion_matrix)){
            best_cm = confusion_matrix;
            best_k = K[k];
        }

        out.open(filename, std::ios::app);
        out << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
        out.close();
    }
    if (fine_tune){
        out.open(filename, std::ios::app);
        out << "Best k : " << best_k << '\n';
        out << "Best Confusion Matrix : " << '\n';
        out.close();
        best_cm.PrintEvaluation(filename);
        out.open(filename, std::ios::app);
        out << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
        out.close();
    }

    cout << "Best k : " << best_k << '\n';
    cout << "Best Confusion Matrix : " << '\n';
    best_cm.PrintEvaluation();

	return 0;
}
