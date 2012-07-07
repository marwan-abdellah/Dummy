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

#include "Utils.h"

int Utils::stringToInt(string_t string)
{
	return atoi(string.c_str());
}

float Utils::stringToFloat(string_t string)
{
	return atof(string.c_str());
}

double Utils::stringToDouble(string_t string)
{
	return atof(string.c_str());
}

string_t Utils::intToString(int intVal)
{
	sstream_t stream;
	stream << (intVal);
	return stream.str();
}

string_t Utils::floatToString(float floatVal)
{
	sstream_t stream;
	stream << (floatVal);
	return stream.str();
}

string_t Utils::doubleToString(double doubleVal)
{
	sstream_t stream;
	stream << (doubleVal);
	return stream.str();
}

string_t Utils::charArrayToString(char* inputCharArray)
{
	string_t string;
	for (int i = 0; i < strlen(inputCharArray); i++)
		string += inputCharArray[i];

	return string;
}

char* Utils::stringToCharArray(string_t inputString)
{

	int stringLen = inputString.size();

	char* outputChar;
	for (int i = 0; i <= stringLen ; i++)
	{
		outputChar[i] = inputString[i];
	}

	return outputChar;
}

string_t Utils::charArrayToString_const (const char* inputCharArray)
{
    string_t string;
    for (int i = 0; i < strlen(inputCharArray); i++)
        string += inputCharArray[i];

    return string;
}

int Utils::rand_int()
{
	return rand();
}

float Utils::rand_float()
{
	return (float) float(rand()) / RAND_MAX;
}

double Utils::rand_double()
{
	return (double) (double(drand48()) / RAND_MAX);
}

int Utils::rand_int_range(int minNum, int maxNum)
{
	 return( rand() % (maxNum - minNum) + minNum);
}
