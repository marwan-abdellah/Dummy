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
