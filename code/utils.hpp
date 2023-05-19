#include <string.h>
#include <random>
#include <ctime>
#include "confusion_matrix/confusion_matrix.hpp"

class RandomNumberGenerator {
public:
    RandomNumberGenerator();

    int generateRandomInteger(int min, int max) ;

    double generateRandomReal(double min, double max) ;

private:
    std::default_random_engine randomEngine;
};

std::string generateFileName(char*);


bool isBetter(ConfusionMatrix& matrix1, ConfusionMatrix& matrix2);