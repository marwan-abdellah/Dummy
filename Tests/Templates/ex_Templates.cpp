/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : Volume
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#include "ex_Templates.h"
#include <iostream>
#include <typeinfo>
#include <iostream>

template <typename T>
void ex_Templates::streamOut(const T val)
{
	std::cout << val << std::endl;

	// Static type checking
	if (typeid(val) == typeid(int))
		std::cout << "int" << std::endl;
	if (typeid(val) == typeid(float))
		std::cout << "float" << std::endl;
	if (typeid(val) == typeid(long))
		std::cout << "long" << std::endl;
	if (typeid(val) == typeid(double))
		std::cout << "double" << std::endl;
}

template void ex_Templates::streamOut <int> (const int val);
template void ex_Templates::streamOut <float> (const float val);
template void ex_Templates::streamOut <long> (const long val);
template void ex_Templates::streamOut <double> (const double val);
