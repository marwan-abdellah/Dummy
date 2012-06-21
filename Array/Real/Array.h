/*
 * Array.h
 *
 *  Created on: May 31, 2012
 *      Author: abdellah
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <fftw3.h>

namespace Array
{
	void fillArray_1D_float(float* arr, int size_X, bool Seq_Rnd);
	void fillArray_2D_flat_float(float* arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_flat_float(float* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	void fillArray_2D_float(float** arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_float(float*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);

	void fillArray_1D_double(double* arr, int size_X, bool Seq_Rnd);
	void fillArray_2D_flat_double(double* arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_flat_double(double* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	void fillArray_2D_double(double** arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_double(double*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
}


#endif /* ARRAY_H_ */
