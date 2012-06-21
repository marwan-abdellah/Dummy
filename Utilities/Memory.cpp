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
				free(ptrData[i][j]);
		free(ptrData[i]);
	}

	free(ptrData);
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
