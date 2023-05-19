#include "confusion_matrix.hpp"
#include <iostream>
#include <iomanip>

using namespace std;

ConfusionMatrix::ConfusionMatrix() {
    // Populate 2x2 matrix with 0s
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            m_confusion_matrix[i][j] = 0;
        }
    }
}

ConfusionMatrix::~ConfusionMatrix() {
    // Attribute m_confusion_matrix is deleted automatically
}

void ConfusionMatrix::AddPrediction(int true_label, int predicted_label) {
    m_confusion_matrix[true_label][predicted_label]++;
}

void ConfusionMatrix::PrintEvaluation(string file) const{
    // Prints the confusion matrix
    ofstream out(file, ios::app);

    // Vérifier si le fichier est ouvert avec succès
    if (!out) {
        cerr << "Erreur lors de l'ouverture du fichier." << '\n';
        return ;
    }

    // Largeur de chaque colonne
    int columnWidths[3] = {13, 8, 8};

    // Écrire les en-têtes de colonnes
    out << setw(columnWidths[0]) << "" << setw(3)  << "" << "Predicted" << '\n';
    out << setw(columnWidths[0]) << "" << setw(columnWidths[1]) << "0" << setw(columnWidths[2]) << "1" << '\n';

    // Écrire les données
    out << setw(columnWidths[0]) << "Actual    0";
    out << setw(columnWidths[1]) << m_confusion_matrix[0][0] << setw(columnWidths[2]) << m_confusion_matrix[0][1] << '\n';
    out << setw(columnWidths[0]) << "          1";
    out << setw(columnWidths[2]) << m_confusion_matrix[1][0] << setw(columnWidths[2]) << m_confusion_matrix[1][1] << '\n';

    // Afficher les estimateurs
    out <<"\nError rate\t\t"
        <<error_rate() << '\n';
    out <<"False alarm rate\t"
        <<false_alarm_rate() << '\n';
    out <<"Detection rate\t\t"
        <<detection_rate() << '\n';
    out <<"F-score\t\t\t"
        <<f_score() << '\n';
    out <<"Precision\t\t"
        <<precision() << '\n';
    // Fermer le fichier
    out.close();
}

void ConfusionMatrix::PrintEvaluation() const{
    // Prints the confusion matrix
    cout <<"\t\tPredicted\n";
    cout <<"\t\t0\t1\n";
    cout <<"Actual\t0\t"
        <<GetTN() <<"\t"
        <<GetFP() << '\n';
    cout <<"\t1\t"
        <<GetFN() <<"\t"
        <<GetTP() <<endl << '\n';
    // Prints the estimators
    cout <<"Error rate\t\t"
        <<error_rate() << '\n';
    cout <<"False alarm rate\t"
        <<false_alarm_rate() << '\n';
    cout <<"Detection rate\t\t"
        <<detection_rate() << '\n';
    cout <<"F-score\t\t\t"
        <<f_score() << '\n';
    cout <<"Precision\t\t"
        <<precision() << '\n';
}

int ConfusionMatrix::GetTP() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix::GetTN() const {
   return m_confusion_matrix[0][0];
}

int ConfusionMatrix::GetFP() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix::GetFN() const {
   return m_confusion_matrix[1][0];
}

double ConfusionMatrix::f_score() const {
    double p = precision();
    double r = detection_rate();
    if (p < 0 || r < 0) return -1;
    return 2 * p * r / (p + r);
}

double ConfusionMatrix::precision() const {
    double den = GetTP() + GetFP();
    if (den == 0) return -1;
    return (double)GetTP() / den;
}

double ConfusionMatrix::error_rate() const {
    return (double)(GetFP() + GetFN()) / 
        (double)(GetFP() + GetFN() + GetTP() + GetTN());
}

double ConfusionMatrix::detection_rate() const {
    double den = GetTP() + GetFN();
    if (den == 0) return -1;
    return (double)GetTP() / den;
}

double ConfusionMatrix::false_alarm_rate() const {
   return (double)GetFP() / (double)(GetFP() + GetTN());
}
