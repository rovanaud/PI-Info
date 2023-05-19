EIGEN=../eigen-3.4.0
ANN_INCLUDE=../ann_1.1.2/include
ANN_LIB=../ann_1.1.2/lib

# LIB = -I$(EIGEN) -I$(ANN_INCLUDE)
INCLUDES = -Idataset -Iconfusion_matrix -Ikernel -Isvm -Iknn 
CXX = g++
CXXFLAGS = ${INCLUDES} -std=c++11 -O2


.PHONY: all clean

all: run 

OBJECTS = utils.o dataset.o confusion_matrix.o svm.o kernel.o Classification.o KnnClassification.o RandomProjection.o test_knn.o test_svm.o test_random_projection.o main.o 
SOURCES = utils.cpp dataset/dataset.cpp confusion_matrix/confusion_matrix.cpp svm/svm.cpp kernel/kernel.cpp knn/Classification.cpp knn/KnnClassification.cpp knn/RandomProjection.cpp knn/test_knn.cpp svm/test_svm.cpp knn/test_random_projection.cpp main.cpp

$(OBJECTS): $(SOURCES)
	$(CXX) $(CXXFLAGS) -c $^ -I$(EIGEN) -I$(ANN_INCLUDE) -L$(ANN_LIB) -lANN

run: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o run -I$(EIGEN) -I$(ANN_INCLUDE)  $(OBJECTS) -L$(ANN_LIB) -lANN

clean:
	rm -f run *.o