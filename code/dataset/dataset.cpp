
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include "dataset.hpp"


int Dataset::GetNbrSamples() const {
	return m_nsamples;
}

int Dataset::GetDim() const {
	return m_dim;
}

Dataset::~Dataset() {
}

void Dataset::Show(bool verbose) const {
	std::cout<<"Dataset with "<<m_nsamples<<" samples, and "<<m_dim<<" dimensions."<<std::endl;

	if (verbose) {
		for (int i=0; i<m_nsamples; i++) {
			for (int j=0; j<m_dim; j++) {
				std::cout<<m_instances[i][j]<<" ";
			}
			std::cout<<std::endl;		
		}
	}
}

Dataset::Dataset(const char* file, double fraction) {
	m_nsamples = 0;
	m_dim = -1;

	std::ifstream fin(file);
	
	if (fin.fail()) {
		std::cout<<"Cannot read from file "<<file<<" !"<<std::endl;
		exit(1);
	}
	
	std::vector<double> row; 
    std::string line, word, temp; 

	while (getline(fin, line)) {
		row.clear();
        std::stringstream s(line);
        
        int valid_sample = 1;
        int ncols = 0;
        while (getline(s, word, ',')) { 
            // add all the column data 
            // of a row to a vector 
            double val = std::atof(word.c_str());
            row.push_back(val);
            ncols++;
        }
        if (!valid_sample) {
            // in this version, valid_sample is always 1
        	continue;
        }         
        m_instances.push_back(row);
        if (m_dim==-1) m_dim = ncols;
        else if (m_dim!=ncols) {
        	std::cerr << "ERROR, inconsistent dataset" << std::endl;
        	exit(-1);
        }
        
		m_nsamples ++;
	}

	fin.close();

	std::random_device rd;
    std::mt19937 gen(rd());

	int m = static_cast<int>((1 - fraction) * m_nsamples);
    if (m >= m_nsamples) {
        // Delete all elements if m is greater than or equal to n
        m_instances.clear();
		m_nsamples = 0;
        return;
    }

    std::vector<int> indices(m_nsamples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    indices.resize(m);  // Keep only the first m random indices

    std::sort(indices.begin(), indices.end(), std::greater<int>()); // Sort in descending order

    for (int index : indices) {
        m_instances.erase(m_instances.begin() +  index);
    }
	
	m_nsamples -= m;
	
}

Dataset::Dataset(const std::vector<std::vector<double> > &input, double fraction) {
	m_nsamples = static_cast<int>(fraction * input.size());;
	
	m_instances = copyRandomSample(input, m_nsamples);

	m_dim = input[0].size();

}

const std::vector<double>& Dataset::GetInstance(int i) const {
	return m_instances[i];
}


std::vector<std::vector<double> > copyRandomSample(const std::vector<std::vector<double> >& input, int size) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<std::vector<double>> shuffled = input;
    std::shuffle(shuffled.begin(), shuffled.end(), g);

    std::vector<std::vector<double> > result;
    result.reserve(size);
    
    std::copy(shuffled.begin(), shuffled.begin() + size, std::back_inserter(result));

    return result;
}