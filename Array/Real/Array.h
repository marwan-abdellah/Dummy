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
	void zeroArray_2D_flat_float(float* arr, int size_X, int size_Y);
	void zeroArray_2D_flat_double(double* arr, int size_X, int size_Y);
	void zeroArray_3D_flat_float(float* arr, int size_X, int size_Y, int size_Z);
	void zeroArray_3D_flat_double(double* arr, int size_X, int size_Y, int size_Z);

	void zeroArray_2D_float(float** arr, int size_X, int size_Y);
	void zeroArray_2D_double(double** arr, int size_X, int size_Y);
	void zeroArray_3D_float(float*** arr, int size_X, int size_Y, int size_Z);
	void zeroArray_3D_double(double*** arr, int size_X, int size_Y, int size_Z);

	void fillArray_1D_int(int* arr, int size_X, bool Seq_Rnd);
	void fillArray_2D_flat_int(int* arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_flat_int(int* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	void fillArray_2D_int(int** arr, int size_X, int size_Y, bool Seq_Rnd);
	void fillArray_3D_int(int*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);

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

	void flatArray_2D_char(char* flatArr, char** sqArr, int size_X, int size_Y);
	void flatArray_2D_float(float* flatArr, float** sqArr, int size_X, int size_Y);
	void flatArray_2D_double(double* flatArr, double** sqArr, int size_X, int size_Y);

	void flatArray_3D_char(char* flatArr, char*** sqArr, int size_X, int size_Y, int size_Z);
	void flatArray_3D_float(float* flatArr, float*** sqArr, int size_X, int size_Y, int size_Z);
	void flatArray_3D_double(double* flatArr, double*** sqArr, int size_X, int size_Y, int size_Z);

	void squareArray_2D_char(char* flatArr, char** sqArr, int size_X, int size_Y);
	void squareArray_2D_float(float* flatArr, float** sqArr, int size_X, int size_Y);
	void squareArray_2D_double(double* flatArr, double** sqArr, int size_X, int size_Y);

	void cubeArray_3D_char(char* flatArr, char*** sqArr, int size_X, int size_Y, int size_Z);
	void cubeArray_3D_float(float* flatArr, float*** sqArr, int size_X, int size_Y, int size_Z);
	void cubeArray_3D_double(double* flatArr, double*** sqArr, int size_X, int size_Y, int size_Z);




	template <typename T>
	extern void zeroArray_2D_flat(T* arr, int size_X, int size_Y);




}


#endif /* ARRAY_H_ */
