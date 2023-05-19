#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "test_random_projection.hpp"
#include "RandomProjection.hpp"
#include "KnnClassification.hpp"
#include "../confusion_matrix/confusion_matrix.hpp"
#include "../utils.hpp"

using namespace std;

/** @file
*/

// Prints correct usage
void msgleave_random_projection(char* argv[]) {
        std::cout << "Usage: " << argv[0] << " <train_file> <test_file> <fine_tune> | [<k> <Dim[d]>  <sampling>] " << '\n';
}

int test_random_projection(int argc, char* argv[]) {
    // Tests correct usage
	if (argc < 6 && argc != 4) {
		msgleave_random_projection(argv);
		return 1;
	}
		
    // Puts train file in a Dataset object
    cout << "Reading Datasets ... " << '\n';
    clock_t t_random_projection = clock();
	Dataset train_dataset(argv[1]);
	Dataset class_dataset(argv[2]);
    t_random_projection = clock() - t_random_projection;
    std::cout << endl
        <<"Execution time: "
        <<(t_random_projection*1000)/CLOCKS_PER_SEC
        <<"ms\n\n";

    int *Dim;
    int *K;
    int fine_tune = 1;
    string* sampling;

    if(argc == 4) {
        RandomNumberGenerator rng;
        fine_tune = atoi(argv[3]);
        Dim = new int[fine_tune];
        K = new int[fine_tune];
        for (int i = 0; i < fine_tune; i++) {
            Dim[i] = rng.generateRandomInteger(1, 767);
            K[i] = rng.generateRandomInteger(1, 10);
        }
        sampling = new string[2];
        sampling[0] = "Gaussian";
        sampling[1] = "Rademacher";
    } else { 
        K = new int[1] ; *K = atoi(argv[3]);
        Dim = new int[1] ; *Dim = atoi(argv[4]);
        sampling = new string[1]; *sampling = argv[5];
    }

    int good_k = -1;
    int good_dim = -1;
    string good_sampling = "empty";


	train_dataset.Show(false);  // only dim and samples
    ConfusionMatrix best_cm;

	// Random projection
    std::cout << "Performing Random Projection fine tune : " << fine_tune <<"\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
    
    

    string filename = generateFileName(argv[0]);

    for (auto d = 0; d < fine_tune; d++) {
        for (auto k = 0; k < fine_tune; k++) {
            for (int s = 0; s < min(2, fine_tune); s++){
                
                ofstream out(filename, std::ios::app);
                out << "Params : dim: " << Dim[d] << " k: " << K[k] << " sampling: " << sampling[s] << '\n';
                cout << "dim: " << Dim[d] << " k: " << K[k] << " sampling: " << sampling[s] << '\n';
                // continue;
                clock_t t_random_projection = clock();
                RandomProjection projection(train_dataset.GetDim()-1, 0, Dim[0], sampling[s]);
                t_random_projection = clock() - t_random_projection;
                
                out << "Projection donne in  : " << (t_random_projection * 1000) / CLOCKS_PER_SEC << "ms" << '\n';
                std::cout << endl
                    <<"Execution time: "
                    <<(t_random_projection*1000)/CLOCKS_PER_SEC
                    <<"ms\n\n";

                
                clock_t t_knn_train_projected = clock();
                Dataset projection_dataset = projection.Project(&train_dataset);
                // Class label is last column
                KnnClassification knn_class_projected(K[k], &projection_dataset, Dim[d]);
                t_knn_train_projected = clock() - t_knn_train_projected;
                std::cout <<"Performing Knn on projected data Execution time: "
                    <<(t_knn_train_projected*1000)/CLOCKS_PER_SEC
                    <<"ms\n";
                out << "Performing Knn on projected data Execution time: " << (t_knn_train_projected*1000)/CLOCKS_PER_SEC << "ms" << '\n';
            
                ConfusionMatrix confusion_matrix_projected;
                Dataset projection_test_dataset = projection.Project(&class_dataset);
                clock_t t_knn_test_projected = clock();
                    for (int i=0; i<projection_test_dataset.GetNbrSamples(); i++) {
                    std::vector<double> sample = projection_test_dataset.GetInstance(i);
                    ANNpoint query = annAllocPt(projection_test_dataset.GetDim()-1, 0);
                    int true_label = -1;  // To not leave it uninitialized + will error in AddPrediction
                    for (int j=0, j2=0; j<projection_test_dataset.GetDim() && j2<projection_test_dataset.GetDim(); j++, j2++) {
                        if (j==Dim[d] && j2==Dim[d]) {
                            true_label = sample[j2];
                            j--;
                            continue;
                        }
                        query[j] = sample[j2];
                    }
                    int predicted_label = knn_class_projected.Estimate(query);
                    annDeallocPt(query);
                    confusion_matrix_projected.AddPrediction(true_label, predicted_label);
                }
                t_knn_test_projected = clock() - t_knn_test_projected;

                std::cout <<"Execution time: " <<(t_knn_test_projected*1000)/CLOCKS_PER_SEC <<"ms\n";

                out << "Testing Knn on projected data Execution time: " << (t_knn_test_projected*1000)/CLOCKS_PER_SEC << "ms" << '\n';
                

                if (isBetter(best_cm, confusion_matrix_projected)) {
                    best_cm = confusion_matrix_projected;
                    good_k = K[k];
                    good_dim = Dim[d];
                    good_sampling = sampling[s];
                }
                out.close();

                ofstream of(filename, std::ios::app);
                of << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
                
                
                confusion_matrix_projected.PrintEvaluation(filename);
                confusion_matrix_projected.PrintEvaluation();
            }
        }
    }

    if (fine_tune > 1) cout << "Best Model : " << '\n';
    else cout << "Result : " << '\n';

    ofstream out(filename, std::ios::app);
    out << "Best Model : " << '\n';
    out << "k: " << good_k << " dim: " << good_dim << " sampling: " << good_sampling << '\n';
    best_cm.PrintEvaluation(filename);
    best_cm.PrintEvaluation();
    return 0;
}
