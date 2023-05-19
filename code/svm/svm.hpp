#include <string>
#include <vector>
#include <functional>
// #include "../dataset/dataset.hpp"
#include "../confusion_matrix/confusion_matrix.hpp"
#include "../kernel/kernel.hpp"

#ifndef SVM_HPP
#define SVM_HPP

class SVM {
    private:
        // training dataset
        int col_class;
        std::vector<std::vector<double>> train_features;
        std::vector<int> train_labels;
        // kernel
        Kernel kernel;
        std::vector<std::vector<double>> computed_kernel;
        // estimation result
        void compute_kernel();
        std::vector<double> alpha;
        double beta_0;
        // only consider support inside the margin by at least clipping_epsilon
        const double clipping_epsilon = 0.0000001;
        // consider stopping gradient ascent when the derivative is smaller than stopping_criterion
        const double stopping_criterion = 0.001;

    public:
        // constructor
        SVM() = delete;
        SVM(Dataset* dataset, int col_class, Kernel K=Kernel({0, 0, 0.0, 0.0}));
        // destructor
        ~SVM();
        // only public for test purposes, should be private
        void compute_beta_0(double C=1.0);

        // getters - setters - for test purposes
        int get_col_class() const;
        Kernel get_kernel() const;
        std::vector<int> get_train_labels() const;
        std::vector<std::vector<double>> get_train_features() const;
        std::vector<std::vector<double>> get_computed_kernel() const;
        std::vector<double> get_alphas() const;
        double get_beta_0() const;
        void set_alphas(std::vector<double> alpha);

        // methods
        void train(const double C, const double lr);
        int f_hat(const std::vector<double> x);
        ConfusionMatrix test(const Dataset* test);
};

#endif
