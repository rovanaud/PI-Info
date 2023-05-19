#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>

#include "../kernel/kernel.hpp"
#include "../confusion_matrix/confusion_matrix.hpp"
#include "svm.hpp"

SVM::SVM(Dataset* dataset, int col_class, Kernel K):
    col_class(col_class), kernel(K) {
    train_labels = std::vector<int>(dataset->GetNbrSamples());
    train_features = std::vector<std::vector<double>>(dataset->GetNbrSamples(), std::vector<double>(dataset->GetDim() - 1));
    // Exercise 2: put the correct columns of dataset in train_labels and _features AGAIN!
    // BEWARE: transform 0/1 labels to -1/1
    for (int i = 0; i < dataset->GetNbrSamples(); i++) {
        train_labels[i] = 2 * dataset->GetInstance(i)[col_class] - 1;
        int k = 0;
        for (int j = 0; j < dataset->GetDim(); j++) 
            if (j != col_class) train_features[i][k++] = dataset->GetInstance(i)[j];
    }
    compute_kernel();
}

SVM::~SVM() {
}

void SVM::compute_kernel() {
    const int n = train_features.size();
    alpha = std::vector<double>(n);
    computed_kernel = std::vector<std::vector<double>>(n, std::vector<double>(n));

    // Exercise 2: put y_i y_j k(x_i, x_j) in the (i, j)th coordinate of computed_kernel
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            computed_kernel[i][j] = train_labels[i] * train_labels[j] * kernel.k(train_features[i], train_features[j]);
        }
    }
}

void SVM::compute_beta_0(double C) {
    // count keeps track of the number of support vectors (denoted by n_s)
    int count = 0;
    int n(alpha.size());
    beta_0 = 0.0;
    // Exercise 3
    // Use clipping_epsilon < alpha < C - clipping_epsilon instead of 0 < alpha < C
    for (int s = 0; s < n; s++) 
        if (alpha[s] > clipping_epsilon && alpha[s] < C - clipping_epsilon && ++count){
            beta_0 -= train_labels[s];
            for (int i = 0; i < n; i++)
                // beta_0 += alpha[i] * train_labels[i] * train_labels[s] * computed_kernel[i][s]; // May not work if computer_kernel is not computed
                beta_0 += alpha[i] * train_labels[i] * kernel.k(train_features[i], train_features[s]);
        }
    // This performs 1/n_s
    beta_0 /= count;
}

void SVM::train(const double C, const double lr) {
    // Exercise 4
    // Perform projected gradient ascent
    int n = alpha.size();
    // While some alpha is not clipped AND its gradient is above stopping_criterion
    // (1) Set stop = false
    bool stop = false;
    // (2) While not stop do
    while (!stop) {
        // (2.1) Set stop = true
        stop = true;
        // (2.2) For i = 1 to n do
        for (int i = 0; i < n; i++) {
            // (2.2.1) Compute the gradient - HINT: make good use of computed_kernel
            double gradient = 0.0;
            for (int j = 0; j < n; j++) gradient += alpha[j] * computed_kernel[i][j];
            gradient = 1 - gradient;
            // (2.2.2) Move alpha in the direction of the gradient - eta corresponds to lr (for learning rate)
            alpha[i] += lr * gradient;
            // (2.2.3) Project alpha in the constraint box by clipping it
            double proj = std::max(0.0, std::min(C, alpha[i]));             
            // (2.2.4) Adjust stop if necessary
            stop = (proj != alpha[i] || std::abs(gradient) <= stopping_criterion) && stop;       
            // (2.2.5) Update alpha[i]
            alpha[i] = proj;
        }
    }
    // (3) Compute beta_0

    // Update beta_0
    
    compute_beta_0(C);
}

int SVM::f_hat(const std::vector<double> x) {
    // Exercise 5
    double sum = 0.0;
    for (int i = 0; i < alpha.size(); i++) {
        sum += alpha[i] * train_labels[i] * kernel.k(train_features[i], x);
    }
    return 2 * (sum - beta_0 > 0) - 1;
}

ConfusionMatrix SVM::test(const Dataset* test) {
    // Collect test_features and test_labels and compute confusion matrix
    std::vector<double> test_labels (test->GetNbrSamples());
    // std::vector<double> predicted_labels (test->GetNbrSamples());
    std::vector<std::vector<double>> test_features (test->GetNbrSamples(), std::vector<double>(test->GetDim() - 1));
    ConfusionMatrix cm;

    // Exercise 6
    // Put test dataset in features and labels
    for (int i = 0; i < test->GetNbrSamples(); i++) {
        test_labels[i] = test->GetInstance(i)[col_class];
        int k = 0;
        for (int j = 0; j < test->GetDim(); j++) 
            if (j != col_class) test_features[i][k++] = test->GetInstance(i)[j];
    }
    // Use f_hat to predict and put into the confusion matrix
    for (int i = 0; i < test->GetNbrSamples(); i++) {
        cm.AddPrediction(test_labels[i], (1 + f_hat(test_features[i])) / 2);
    }
    // BEWARE: transform -1/1 prediction to 0/1 label

    return cm;
}

int SVM::get_col_class() const {
    return col_class;
}

Kernel SVM::get_kernel() const {
    return kernel;
}

std::vector<int> SVM::get_train_labels() const {
    return train_labels;
}

std::vector<std::vector<double>> SVM::get_train_features() const {
    return train_features;
}

std::vector<std::vector<double>> SVM::get_computed_kernel() const {
    return computed_kernel;
}

std::vector<double> SVM::get_alphas() const {
    return alpha;
}

double SVM::get_beta_0() const {
    return beta_0;
}

void SVM::set_alphas(std::vector<double> alpha) {
    this->alpha = alpha;
}
