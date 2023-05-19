#include "../dataset/dataset.hpp"

#ifndef KERNEL_HPP
#define KERNEL_HPP

/**
  All SVM parameters.
*/
struct kernel_parameter
{
	int kernel_type;
	int degree;	 /* for poly */
	double gamma;  /* for poly/rbf/sigmoid */
	double coef0;  /* for poly/sigmoid */
};

/**
  Kernel types.
*/
enum {LINEAR, POLY, RBF, SIGMOID, RATQUAD};

/**
  The Kernel class defines the kernel type, its parameters, and computes the kernel.
*/
class Kernel {
    public:
		/**
		 The constructor needs:
		@param kernel_parameter the kernel parameters
		*/
	    Kernel(const kernel_parameter& param);
		/**
		 The kernel function.
		*/
	    double k(const std::vector<double> &x1, const std::vector<double> &x2) const;
		int get_kernel_type() const;

    private:
	    const int kernel_type;
	    const int degree;
	    const double gamma;
	    const double coef0;

	    static double dot(const std::vector<double> &x1, const std::vector<double> &x2);
	    double kernel_linear(const std::vector<double> &x1, const std::vector<double> &x2) const;
	    double kernel_poly(const std::vector<double> &x1, const std::vector<double> &x2) const;
	    double kernel_rbf(const std::vector<double> &x1, const std::vector<double> &x2) const;
	    double kernel_sigmoid(const std::vector<double> &x1, const std::vector<double> &x2) const;
	    double kernel_ratquad(const std::vector<double> &x1, const std::vector<double> &x2) const;
};

#endif
