/*
 * Memory.cpp
 *
 *  Created on: May 23, 2012
 *      Author: abdellah
 */

#include "Memory.h"

char* Memory::alloc_1D_char(const int size_X)
{
	LOG();

	char* data = (char*) malloc (sizeof(char) * size_X);
	return data;
}

float* Memory::alloc_1D_float(const int size_X)
{
	LOG();
	float* data = (float*) malloc (sizeof(float) * size_X);
	return data;
}
double* Memory::alloc_1D_double(const int size_X)
{
	LOG();
	double* data = (double*) malloc (sizeof(double) * size_X);
	return data;
}

char** Memory::alloc_2D_char(const int size_X, const int size_Y)
{
	LOG();

	char** data;
	data = (char**) malloc (sizeof(char*) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (char*) malloc (sizeof(char) * size_Y);

	return data;
}

float** Memory::alloc_2D_float(const int size_X, const int size_Y)
{
	LOG();

	float** data;
	data = (float**) malloc (sizeof(float*) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (float*) malloc (sizeof(float) * size_Y);

	return data;
}
double** Memory::alloc_2D_double(const int size_X, const int size_Y)
{
	LOG();

	double** data;
	data = (double**) malloc (sizeof(double) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (double*) malloc (sizeof(double) * size_Y);

	return data;
}

char*** Memory::alloc_3D_char(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	char*** data;
	data = (char***) malloc (sizeof(char**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (char**) malloc (sizeof(char*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (char*) malloc (sizeof(char) * size_Z);
	}

	return data;
}

float*** Memory::alloc_3D_float(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	float*** data;
	data = (float***) malloc (sizeof(float**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (float**) malloc (sizeof(float*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (float*) malloc (sizeof(float) * size_Z);
	}

	return data;
}

double*** Memory::alloc_3D_double(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	double*** data;
	data = (double***) malloc (sizeof(double**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (double**) malloc (sizeof(double*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (double*) malloc (sizeof(double) * size_Z);
	}

	return data;
}

cufftComplex** Memory::alloc_2D_cufftComplex(const int size_X, const int size_Y)
{
	LOG();

	cufftComplex** data;
	data = (cufftComplex**) malloc (sizeof(cufftComplex) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (cufftComplex*) malloc (sizeof(cufftComplex) * size_Y);

	return data;
}

cufftDoubleComplex** Memory::alloc_2D_cufftDoubleComplex(const int size_X, const int size_Y)
{
	LOG();

	cufftDoubleComplex** data;
	data = (cufftDoubleComplex**) malloc (sizeof(cufftDoubleComplex) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (cufftDoubleComplex*) malloc (sizeof(cufftDoubleComplex) * size_Y);

	return data;
}

cufftComplex*** Memory::alloc_3D_cufftComplex(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	cufftComplex*** data;
	data = (cufftComplex***) malloc (sizeof(cufftComplex**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (cufftComplex**) malloc (sizeof(cufftComplex*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (cufftComplex*) malloc (sizeof(cufftComplex) * size_Z);
	}

	return data;
}

cufftDoubleComplex*** Memory::alloc_3D_cufftDoubleComplex(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	cufftDoubleComplex*** data;
	data = (cufftDoubleComplex***) malloc (sizeof(cufftDoubleComplex**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (cufftDoubleComplex**) malloc (sizeof(cufftDoubleComplex*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (cufftDoubleComplex*) malloc (sizeof(cufftDoubleComplex) * size_Z);
	}

	return data;
}

fftwf_complex** Memory::alloc_2D_fftwfComplex(const int size_X, const int size_Y)
{
	LOG();

	fftwf_complex** data;
	data = (fftwf_complex**) malloc (sizeof(fftwf_complex) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (fftwf_complex*) malloc (sizeof(fftwf_complex) * size_Y);

	return data;
}

fftw_complex** Memory::alloc_2D_fftwComplex(const int size_X, const int size_Y)
{
	LOG();

	fftw_complex** data;
	data = (fftw_complex**) malloc (sizeof(fftw_complex) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (fftw_complex*) malloc (sizeof(fftw_complex) * size_Y);

	return data;
}


fftwf_complex*** Memory::alloc_3D_fftwfComplex(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	fftwf_complex*** data;
	data = (fftwf_complex***) malloc (sizeof(fftwf_complex**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (fftwf_complex**) malloc (sizeof(fftwf_complex*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (fftwf_complex*) malloc (sizeof(fftwf_complex) * size_Z);
	}

	return data;
}

fftw_complex*** Memory::alloc_3D_fftwComplex(const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	fftw_complex*** data;
	data = (fftw_complex***) malloc (sizeof(fftw_complex**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (fftw_complex**) malloc (sizeof(fftw_complex*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (fftw_complex*) malloc (sizeof(fftw_complex) * size_Z);
	}

	return data;
}

void Memory::free_2D_float(float** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_double(double** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_float(float*** ptrData, const int size_X, const int size_Y, const int size_Z)
{
    for(int i = 0; i < size_Y; i++)
    {
        for(int j = 0; j < size_Z; j++)
                free((void*) ptrData[i][j]);
        free((void*)ptrData[i]);
    }

    free((void*) ptrData);
    ptrData = NULL;
}

void Memory::free_3D_char(char*** ptrData, const int size_X, const int size_Y, const int size_Z)
{
    for(int i = 0; i < size_Y; i++)
    {
        for(int j = 0; j < size_Z; j++)
                free((void*) ptrData[i][j]);
        free((void*)ptrData[i]);
    }

    free((void*) ptrData);
    ptrData = NULL;
}

void Memory::free_3D_double(double*** ptrData, const int size_X, const int size_Y, const int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_cufftComplex(cufftComplex** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_cufftDoubleComplex(cufftDoubleComplex** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_cufftComplex(cufftComplex*** ptrData, const int size_X, const int size_Y, int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_cufftDoubleComplex(cufftDoubleComplex*** ptrData, const int size_X, const int size_Y, int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_fftwfComplex(fftwf_complex** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_fftwComplex(fftw_complex** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_fftwfComplex(fftwf_complex*** ptrData, const int size_X, const int size_Y, int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_fftwComplex(fftw_complex*** ptrData, const int size_X, const int size_Y, int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}








template <typename T>
T* Memory::alloc_1D(const int size_X)
{
	T* data = (T*) malloc (sizeof(T) * size_X);
	return data;
}

template <typename T>
T** Memory::alloc_2D(const int size_X, const int size_Y)
{
	T** data;
	data = (T**) malloc (sizeof(T*) * size_X);
	for(int i = 0; i < size_X; i++)
		data[i] = (T*) malloc (sizeof(T) * size_Y);

	return data;
}

template <typename T>
T*** Memory::alloc_3D(const int size_X, const int size_Y, const int size_Z)
{
	T*** data;
	data = (T***) malloc (sizeof(T**) * size_X);
	for(int i = 0; i < size_X; i++)
	{
		data[i] = (T**) malloc (sizeof(T*) * size_Y);
		for (int j = 0; j < size_Y; j++)
			data[i][j] = (T*) malloc (sizeof(T) * size_Z);
	}

	return data;
}

template <typename T>
void Memory::free_1D(T* ptrData)
{
	free(ptrData);
	ptrData = NULL;
}

template <typename T>
void Memory::free_2D(T** ptrData, const int size_X, const int size_Y)
{
	for (int i = 0; i < size_Y; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

template <typename T>
void Memory::free_3D(T*** ptrData, const int size_X, const int size_Y, int size_Z)
{
	for(int i = 0; i < size_Y; i++)
	{
		for(int j = 0; j < size_Z; j++)
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
	ptrData = NULL;
}

template char* Memory::alloc_1D <char> (const int size_X);
template int* Memory::alloc_1D <int> (const int size_X);
template long* Memory::alloc_1D <long> (const int size_X);
template float* Memory::alloc_1D <float> (const int size_X);
template double* Memory::alloc_1D <double> (const int size_X);
template fftwf_complex* Memory::alloc_1D <fftwf_complex> (const int size_X);
template fftw_complex* Memory::alloc_1D <fftw_complex> (const int size_X);
template cufftComplex* Memory::alloc_1D <cufftComplex> (const int size_X);
template cufftDoubleComplex* Memory::alloc_1D <cufftDoubleComplex> (const int size_X);

template char** Memory::alloc_2D <char> (const int size_X, const int size_Y);
template int** Memory::alloc_2D <int> (const int size_X, const int size_Y);
template long** Memory::alloc_2D <long> (const int size_X, const int size_Y);
template float** Memory::alloc_2D <float> (const int size_X, const int size_Y);
template double** Memory::alloc_2D <double> (const int size_X, const int size_Y);
template fftwf_complex** Memory::alloc_2D <fftwf_complex> (const int size_X, const int size_Y);
template fftw_complex** Memory::alloc_2D <fftw_complex> (const int size_X, const int size_Y);
template cufftComplex** Memory::alloc_2D <cufftComplex> (const int size_X, const int size_Y);
template cufftDoubleComplex** Memory::alloc_2D <cufftDoubleComplex> (const int size_X, const int size_Y);

template void Memory::free_1D <char> (char* ptrData);
template void Memory::free_1D <int> (int* ptrData);
template void Memory::free_1D <long> (long* ptrData);
template void Memory::free_1D <float> (float* ptrData);
template void Memory::free_1D <double> (double* ptrData);
template void Memory::free_1D <fftwf_complex> (fftwf_complex* ptrData);
template void Memory::free_1D <fftw_complex> (fftw_complex* ptrData);
template void Memory::free_1D <cufftComplex> (cufftComplex* ptrData);
template void Memory::free_1D <cufftDoubleComplex> (cufftDoubleComplex* ptrData);

template void Memory::free_2D <char> (char** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <int> (int** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <long> (long** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <float> (float** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <double> (double** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <fftwf_complex> (fftwf_complex** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <fftw_complex> (fftw_complex** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <cufftComplex> (cufftComplex** ptrData, const int size_X, const int size_Y);
template void Memory::free_2D <cufftDoubleComplex> (cufftDoubleComplex** ptrData, const int size_X, const int size_Y);

template char*** Memory::alloc_3D <char> (const int size_X, const int size_Y, const int size_Z);
template int*** Memory::alloc_3D <int> (const int size_X, const int size_Y, const int size_Z);
template long*** Memory::alloc_3D <long> (const int size_X, const int size_Y, const int size_Z);
template float*** Memory::alloc_3D <float> (const int size_X, const int size_Y, const int size_Z);
template double*** Memory::alloc_3D <double> (const int size_X, const int size_Y, const int size_Z);
template fftwf_complex*** Memory::alloc_3D <fftwf_complex> (const int size_X, const int size_Y, const int size_Z);
template fftw_complex*** Memory::alloc_3D <fftw_complex> (const int size_X, const int size_Y, const int size_Z);
template cufftComplex*** Memory::alloc_3D <cufftComplex> (const int size_X, const int size_Y, const int size_Z);
template cufftDoubleComplex*** Memory::alloc_3D <cufftDoubleComplex> (const int size_X, const int size_Y, const int size_Z);

template void Memory::free_3D <char> (char*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <int> (int*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <long> (long*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <float> (float*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <double> (double*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <fftwf_complex> (fftwf_complex*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <fftw_complex> (fftw_complex*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <cufftComplex> (cufftComplex*** ptrData, const int size_X, const int size_Y, int size_Z);
template void Memory::free_3D <cufftDoubleComplex> (cufftDoubleComplex*** ptrData, const int size_X, const int size_Y, int size_Z);
