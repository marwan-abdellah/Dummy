/*
 * Memory.h
 *
 *  Created on: May 23, 2012
 *      Author: abdellah
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include <cstdlib>
#include "LoggingMACROS.h"
#include <fftw3.h>
#include <cufft.h>

namespace Memory
{
	char* alloc_1D_char(const int size_X);
	float* alloc_1D_float(const int size_X);
	double* alloc_1D_double(const int size_X);

	char** alloc_2D_char(const int size_X, const int size_Y);
	float** alloc_2D_float(const int size_X, const int size_Y);
	double** alloc_2D_double(const int size_X, const int size_Y);

	char*** alloc_3D_char(const int size_X, const int size_Y, const int size_Z);
	float*** alloc_3D_float(const int size_X, const int size_Y, const int size_Z);
	double*** alloc_3D_double(const int size_X, const int size_Y, const int size_Z);

	cufftComplex** alloc_2D_cufftComplex(const int size_X, const int size_Y);
	cufftDoubleComplex** alloc_2D_cufftDoubleComplex(const int size_X, const int size_Y);

	cufftComplex*** alloc_3D_cufftComplex(const int size_X, const int size_Y, const int size_Z);
	cufftDoubleComplex*** alloc_3D_cufftDoubleComplex(const int size_X, const int size_Y, const int size_Z);

	fftwf_complex** alloc_2D_fftwfComplex(const int size_X, const int size_Y);
	fftw_complex** alloc_2D_fftwComplex(const int size_X, const int size_Y);

	fftwf_complex*** alloc_3D_fftwfComplex(const int size_X, const int size_Y, const int size_Z);
	fftw_complex*** alloc_3D_fftwComplex(const int size_X, const int size_Y, const int size_Z);

	void free_1D_char(char* ptrData);
	void free_1D_float(float* ptrData);
	void free_1D_double(double* ptrData);

	void free_2D_char(char** ptrData, const int size_X, const int size_Y);
	void free_2D_float(float** ptrData, const int size_X, const int size_Y);
	void free_2D_double(double** ptrData, const int size_X, const int size_Y);

	void free_3D_char(char*** ptrData, const int size_X, const int size_Y, const int size_Z);
	void free_3D_float(float*** ptrData, const int size_X, const int size_Y, const int size_Z);
	void free_3D_double(double*** ptrData, const int size_X, const int size_Y, const int size_Z);

	void free_2D_cufftComplex(cufftComplex** ptrData, const int size_X, const int size_Y);
	void free_2D_cufftDoubleComplex(cufftDoubleComplex** ptrData, const int size_X, const int size_Y);

	void free_3D_cufftComplex(cufftComplex*** ptrData, const int size_X, const int size_Y, int size_Z);
	void free_3D_cufftDoubleComplex(cufftDoubleComplex*** ptrData, const int size_X, const int size_Y, int size_Z);

	void free_2D_fftwfComplex(fftwf_complex** ptrData, const int size_X, const int size_Y);
	void free_2D_fftwComplex(fftw_complex** ptrData, const int size_X, const int size_Y);

	void free_3D_fftwfComplex(fftwf_complex*** ptrData, const int size_X, const int size_Y, int size_Z);
	void free_3D_fftwComplex(fftw_complex*** ptrData, const int size_X, const int size_Y, int size_Z);

	template <typename T>
	extern T* alloc_1D(const int size_X);

	template <typename T>
	extern T** alloc_2D(const int size_X, const int size_Y);

	template <typename T>
	extern T*** alloc_3D(const int size_X, const int size_Y, const int size_Z);

	template <typename T>
	void free_1D(T* ptrData);

	template <typename T>
	void free_2D(T** ptrData, const int size_X, const int size_Y);

	template <typename T>
	void free_3D(T*** ptrData, const int size_X, const int size_Y, const int size_Z);









}

#endif /* MEMORY_H_ */
