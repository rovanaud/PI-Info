
#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>


KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
: Classification(dataset, col_class), m_k(k) {
    // TODO Exercise 1

    int n = dataset->GetNbrSamples();
    int d = dataset->GetDim();
    // std::cout << "dimension : " << d <<  " -- col_class : " << col_class << std::endl;
    m_dataPts = new ANNpoint[n];
    for (int i = 0; i < n; i++) {
        m_dataPts[i] = new double[d-1];
        int cur = 0;
        for(int j = 0; j < d; j++) 
            if (j == col_class) continue;
            else m_dataPts[i][cur++] = dataset->GetInstance(i)[j];
    }
    m_kdTree = new ANNkd_tree(m_dataPts, n, d-1);
}

KnnClassification::~KnnClassification() {
    // TODO Exercise 1
    delete m_kdTree;
    delete[] m_dataPts;

}

int KnnClassification::Estimate(const ANNpoint &x, double threshold) const {
    // TODO Exercise 2
    int knn[m_k];
    double knn_dist[m_k];

    m_kdTree-> annkSearch(x, m_k, knn, knn_dist, threshold);

    /* Methode 1*/
    // int freq{0}; /* value : 0 / 1*/
    // for (int i = 0; i < m_k; i++) 
    //     freq += m_dataset->GetInstance(knn[i])[getColClass()];
    
    // return double(freq / m_k) > threshold;
    /* Methode 1*/

    /* Methode 2
     * Here Better than the other
    */
    double freq[2]{0, 0};
    for (int i = 0; i < m_k; ++i) 
        freq[int(m_dataset->GetInstance(knn[i])[getColClass()])] += double(1 / knn_dist[i]);
    
    return freq[0] < freq[1];
    /* Methode 2*/
}

int KnnClassification::getK() const {
    return m_k;
}

ANNkd_tree *KnnClassification::getKdTree() {
    return m_kdTree;
}
