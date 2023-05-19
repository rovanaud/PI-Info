#include "utils.hpp"

RandomNumberGenerator::RandomNumberGenerator() {
    // Utilisez une graine aléatoire pour initialiser le générateur de nombres aléatoires
    randomEngine.seed(std::random_device()());
}

int RandomNumberGenerator::generateRandomInteger(int min, int max) {
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(randomEngine);
}

double RandomNumberGenerator::generateRandomReal(double min, double max) {
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(randomEngine);
}

std::string generateFileName(char *model) {
    std::time_t now = std::time(nullptr);
    std::tm* timeInfo = std::localtime(&now);

    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", timeInfo);

    return std::string(model) + std::string(buffer) + ".txt";
}

// compares model performance
// returns true if matrix2 is better than matrix1 or if matrix1 is [[0, 0], [0, 0]]
bool isBetter(ConfusionMatrix& best, ConfusionMatrix& candidate) {
    return best.GetTP() == 0 || (candidate.detection_rate() > best.detection_rate() || candidate.f_score() > best.f_score());
}