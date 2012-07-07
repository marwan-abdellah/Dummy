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

#include "Array.h"
#include "Utilities/MACROS.h"
#include "Utilities/LoggingMACROS.h"

void Array::zeroArray_2D_flat_float(float* arr, int size_X, int size_Y)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			arr[index] = (float) 0;
		}
}

void Array::zeroArray_2D_flat_double(double* arr, int size_X, int size_Y)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			arr[index] = (double) 0;
		}
}

void Array::zeroArray_3D_flat_float(float* arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				arr[ctr++] = (float) 0;

}

void Array::zeroArray_3D_flat_double(double* arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				arr[ctr++] = (double) 0;

}

void Array::fillArray_1D_float(float* arr, int size_X, bool Seq_Rnd)
{

	for (int i = 0; i < size_X; i++)
		if (Seq_Rnd)
			arr[i] = (float) i;
		else
			arr[i] = (float) Utils::rand_float();
}

void Array::fillArray_2D_flat_float(float* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));

			if (Seq_Rnd)
				arr[index] = (float) index;
			else
				arr[index] = (float) Utils::rand_float();
		}
}

void Array::fillArray_3D_flat_float(float* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{

				if (Seq_Rnd)
					arr[ctr] = (float) ctr;
				else
					arr[ctr] = (float) Utils::rand_float();

				ctr++;
			}
}

void Array::zeroArray_2D_float(float** arr, int size_X, int size_Y)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			arr[i][j] = (float) 0;

}

void Array::zeroArray_2D_double(double** arr, int size_X, int size_Y)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			arr[i][j] = (double) 0;

}

void Array::zeroArray_3D_float(float*** arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				arr[i][j][k] = (float) 0;

}

void Array::zeroArray_3D_double(double*** arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				arr[i][j][k] = (double) 0;

}


void Array::fillArray_2D_float(float** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
				arr[i][j] = (float) ctr++;
			else
				arr[i][j] = (float) Utils::rand_float();
		}
}

void Array::fillArray_3D_float(float*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
					arr[i][j][k] = ctr++;
				else
					arr[i][j][k] = (float) Utils::rand_float();
			}
}

void Array::fillArray_1D_double(double* arr, int size_X, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
	{
		if (Seq_Rnd)
			arr[i] = (double) i;
		else
			arr[i] = (double) Utils::rand_double();
	}
}

void Array::fillArray_2D_flat_double(double* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));

			if (Seq_Rnd)
				arr[index] = (double) index;
			else
				arr[index] = (double) Utils::rand_double();
		}
}

void Array::fillArray_2D_flat_int(int* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));

			if (Seq_Rnd)
				arr[index] = (int) index;
			else
				arr[index] = (int) Utils::rand_int_range(-128, 128);
		}
}

void Array::fillArray_3D_flat_double(double* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				int index = (i + (size_X * j) + (size_X * size_Y * k));

				if (Seq_Rnd)
					arr[index] = (double) index;
				else
					arr[index] = (double) Utils::rand_double();
			}
}

void Array::fillArray_2D_double(double** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
				arr[i][j] = (double) ctr++;
			else
				arr[i][j] = (double) Utils::rand_double();
		}
}

void Array::fillArray_3D_double(double*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
					arr[i][j][k] = (double) ctr++;
				else
					arr[i][j][k] = (double) Utils::rand_double();
			}
}

void Array::flatArray_2D_char(char* flatArr, char** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				flatArr[ctr++] = sqArr[i][j];
}

void Array::flatArray_2D_float(float* flatArr, float** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				flatArr[ctr++] = sqArr[i][j];
}

void Array::flatArray_2D_double(double* flatArr, double** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				flatArr[ctr++] = sqArr[i][j];
}

void Array::flatArray_3D_char(char* flatArr, char*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				flatArr[ctr++] = sqArr[i][j][k];
}

void Array::flatArray_3D_float(float* flatArr, float*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				flatArr[ctr++] = sqArr[i][j][k];
}

void Array::flatArray_3D_double(double* flatArr, double*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				flatArr[ctr++] = sqArr[i][j][k];
}


void Array::squareArray_2D_char(char* flatArr, char** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				 sqArr[i][j] = flatArr[ctr++];
}

void Array::squareArray_2D_float(float* flatArr, float** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				 sqArr[i][j] = flatArr[ctr++];
}

void Array::squareArray_2D_double(double* flatArr, double** sqArr, int size_X, int size_Y)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
				 sqArr[i][j] = flatArr[ctr++];
}

void Array::cubeArray_3D_char(char* flatArr, char*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				 sqArr[i][j][k] = flatArr[ctr++];
}

void Array::cubeArray_3D_float(float* flatArr, float*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				 sqArr[i][j][k] = flatArr[ctr++];
}

void Array::cubeArray_3D_double(double* flatArr, double*** sqArr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				 sqArr[i][j][k] = flatArr[ctr++];
}






template <typename T>
void Array::zeroArray_2D_flat (T* arr, int size_X, int size_Y)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));

			// Static type checking
			if (typeid(T) == typeid(int) ||
				typeid(T) == typeid(long) ||
				typeid(T) == typeid(float) ||
				typeid(T) == typeid(double))
			{
				arr[index] = (T) 0;
			}
			else if (typeid(T) == typeid(fftw_complex) ||
					 typeid(T) == typeid(fftw_complex))
			{
				arr[index][0] = (T) 0;
				arr[index][1] = (T) 0;
			}
			else if (typeid(arr) == typeid(fftw_complex) ||
					 typeid(arr) == typeid(fftw_complex))
			{
				arr[index].x = (T) 0;
				arr[index].y = (T) 0;
			}
			else
			{
				INFO ("");
			}
		}
}








