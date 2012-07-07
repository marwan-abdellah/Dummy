/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <typeinfo>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <stdio.h>
#include <string.h>

#include "Typedefs.h"
#include "Globals.h"

namespace Utils
{
	int stringToInt(string_t string);
    int stringToInt_const(const string_t string);
    float stringToFloat(string_t string);
    float stringToFloat_const(const string_t string);
	double stringToDouble(string_t string);
    double stringToDouble_const(const string_t string);

	string_t intToString(int intVal);
    string_t intToString_const(const int intVal);
	string_t floatToString(float floatVal);
    string_t floatToString_const(const float floatVal);
	string_t doubleToString(double doubleVal);
    string_t doubleToString_const(const double doubleVal);
	string_t charArrayToString(char* inputCharArray);
    string_t charArrayToString_const(const char* inputCharArray);

    char* stringToCharArray(string_t inputString);

    int rand_int();
    float rand_float();
    double rand_double();
    int rand_int_range(int minNum, int maxNum);

}

#endif /* _UTILS_H_ */
