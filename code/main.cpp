#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "svm/test_svm.hpp"
#include "knn/test_random_projection.hpp"
#include "knn/test_knn.hpp"

using namespace std;

// Prints correct usage
void msgleave(char* argv[]) {
    cout << "Usage: " << argv[0] << endl;
    cout << "1: " + string(argv[0])  + " knn [args]" << endl;
    cout << "2: " + string(argv[0])  + " svm [args] ]" << endl;
    cout << "3: " + string(argv[0])  + " random_projection [args]]" << endl;
    cout << std::endl;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        msgleave(argv);
        return 1;
    }

    string model(argv[1]);

    cout << "\n\tModel : " << model << endl;

    if (model == "knn"){
        test_knn(argc - 1, argv + 1);
    } else if (model == "svm") {
        test_svm(argc - 1, argv + 1);
    } else if (model == "random_projection"){
        test_random_projection(argc - 1, argv + 1);
    } else {
        cout << "Invalid model" << endl;
    }
    return 0;
}