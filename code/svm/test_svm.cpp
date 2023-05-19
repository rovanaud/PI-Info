#include <iostream>
#include <string>
#include <random>
#include <chrono> // To seed the random generator
#include <cmath>
#include <cassert>
#include <fstream>
#include <ctime>
#include <system_error>

#include "svm.hpp"
#include "../utils.hpp"

using namespace std;

int test_svm(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <train_file> <test_file> [ <fine_tune> | <1-5 : LINEAR, POLY, RBF, SIGMOID, RATQUAD> [params] [C, lr]]" << endl;
        return 1;
    }

    const char* train_file = argv[1];
    const char* test_file = argv[2];

    cout << "Reading Data     ... [/" << endl;
    // Files are too big for SVM
    Dataset train_dataset(train_file, 0.4);
    Dataset test_dataset(test_file, 0.6);
    cout << "Finished Reading Data /]" << endl;

    train_dataset.Show(false);  // only dim and samples
    test_dataset.Show(false);  // only dim and samples

    string filename = generateFileName(argv[0]);
    ofstream out;

    clock_t t;
    int fine_tune = 1;
    if (argc == 4) fine_tune = max(1, atoi(argv[3]));
    else {
        double gamma(0), coef0(0), lr(0.01);
        int degree(0), C(10);
        switch (atoi(argv[3])) {
            case 1: 
                if (argc > 4) C = atoi(argv[4]);
                if (argc > 5) lr = atof(argv[5]);
                break;
            case 2:
                if (argc < 7) {
                    cerr << "Préciser les parametres: " << argv[0] << " <train_file> <test_file> 2 <degree> <coef0> <gamma>" << endl;
                    return 1;
                } else {
                    degree = atoi(argv[4]);
                    coef0 = atof(argv[5]);
                    gamma = atof(argv[6]);
                    if (argc > 7) C = atoi(argv[7]);
                    if (argc > 8) lr = atof(argv[8]);
                }
                break;
            case 3:
                if (argc < 5) {
                    cerr << "Preciser gamma: " << argv[0] << " <train_file> <test_file> 3 <gamma>" << endl;
                    return 1;
                }
                else {
                    gamma = atof(argv[5]);
                }
                if (argc > 5) C = atoi(argv[5]);
                if (argc > 6) lr = atof(argv[6]);
                break;
            case 4:
                if (argc < 6){
                    cerr << "Preciser coef0 et gamma: " << argv[0] << " <train_file> <test_file> 4 <coef0> <gamma>" << endl;
                    return 1;
                }
                else {
                    coef0 = atof(argv[4]);
                    gamma = atof(argv[5]);
                }
                if (argc > 6) C = atoi(argv[6]);
                if (argc > 7) lr = atof(argv[7]);
                break;
            case 5:
                if (argc < 5) {
                    cerr << "Preciser coef0: " << argv[0] << " <train_file> <test_file> 5 <coef0>" << endl;
                    return 1;
                }
                else {
                    coef0 = atof(argv[4]);
                }
                if (argc > 5) C = atoi(argv[5]);
                if (argc > 6) lr = atof(argv[6]);
                break;
            default:
                cerr << "Invalid kernel type" << endl;
                return 1;
        }
        
        string kerns[6] = {"", "LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD"};
        cout << "Kernel : " << kerns[atoi(argv[3])] << " Gamma : " << gamma << " coef0 : " << coef0 << " degree : " << degree << endl;
        
        
        Kernel k({atoi(argv[3]), degree, gamma, coef0});
        cout << "Training and Testing SVM" << endl;
        t = clock();
        SVM svm(&train_dataset, 0, k);
        svm.train(C, lr);
        cout << "Done : " << (t * 1000) / CLOCKS_PER_SEC << "ms" << endl;
        cout << "Testing SVM " << endl;
        t = clock();
        ConfusionMatrix cm = svm.test(&test_dataset);
        t = clock() - t;

        
        out.open(filename);
        out << "Training and Testing SVM" << endl;
        out << " Execution time : " << (t * 1000) / CLOCKS_PER_SEC << "ms" << endl;
        out << "Kernel : " << kerns[atoi(argv[3])] << " Gamma : " << gamma << " coef0 : " << coef0 << " degree : " << degree << endl;
        
        out << "C : " << C << " lr : " << lr << endl;
        out.close();
        cm.PrintEvaluation(generateFileName(argv[0]));
        cm.PrintEvaluation();
        return 0;
    }
        

    int col_class = 0;

    int degree[fine_tune];
    double coef0[fine_tune];
    double gamma[fine_tune];

    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();

    double upper_bound = 1.0 / std::sqrt(fine_tune);
    double lower_bound = -upper_bound;
    
    std::uniform_real_distribution<double>
        unif(lower_bound, upper_bound);

    std::uniform_int_distribution<int> unif_int(0, 10);

    std::default_random_engine re;

    // obtain a seed from the timer
    myclock::duration d = myclock::now() - beginning;
    re.seed(d.count());

    for (int i = 0; i < fine_tune; i++){ 
        degree[i] = unif_int(re);
        coef0[i] = unif(re);
        gamma[i] = unif(re);
    }

    int C[3] = {4, 8, 16};
    double learning_rate[2] = {0.001, 0.01};

    ConfusionMatrix confusion_matrix;
    std:: cout << "Training and fine_tune" << std::endl;
    int good_type = 0;
    int good_deg = -1;
    double good_coef = -1;
    double good_gamma = -1;
    int good_C = -1;

    t = clock();
    for (int type = 1; type < 5; type++){
        for (auto deg : degree) {
            for (auto coef : coef0) {
                for (auto gam : gamma) {
                    for (auto c : C) {
                        for (auto lr :learning_rate){
                            Kernel k({type, deg, coef, gam});
                            SVM svm(&train_dataset, 0, k);
                            svm.train(c, lr);
                            ConfusionMatrix cm = svm.test(&test_dataset);

                            out.open(filename, std::ios::app);
                            out << "type = " << type << " deg = " << deg << " coef " << coef << " gamma = " << gam << " C = " << c << " lr = " << lr << '\n';
                            out.close();

                            cm.PrintEvaluation(filename);

                            out.open(filename, std::ios::app);
                            out << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n'; 
                            out.close();
                            cout << "type = " << type << " deg = " << deg << " coef " << coef << " gamma = " << gam << " C = " << c << " lr = " << lr << '\n';
                            cout << "TP: " << cm.GetTP() << '\n';
                            cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n'; 
                            if (isBetter(confusion_matrix, cm)){
                                confusion_matrix = cm;
                                good_type = type;
                                good_deg = deg;
                                good_coef = coef;
                                good_gamma = gam;
                                good_C = c;
                            }
                        }
                    }
                }
            }
        }
    }

    t = clock() - t;

    out <<endl
         <<"Execution time: "
         <<(t*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    cout <<endl
         <<"Execution time: "
         <<(t*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    out << "Best type = " << good_type << " def = " << good_deg << " coef " << good_coef << " gamma = " << good_gamma << "C = " << good_C << '\n';
    cout << "Best type = " << good_type << " def = " << good_deg << " coef " << good_coef << " gamma = " << good_gamma << "C = " << good_C << '\n';
    out.close();

    confusion_matrix.PrintEvaluation(filename);

    return 0;
}