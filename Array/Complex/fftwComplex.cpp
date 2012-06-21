/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : fftwComplex.h
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#include "fftwComplex.h"

void Array::fftwComplex::fillArray_1D(fftwf_complex* arr, int size_X, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
	{
		if (Seq_Rnd)
		{
			arr[i][0] = (float) i;
			arr[i][1] = (float) i;
		}
		else
		{
			arr[i][0] = (float) Utils::rand_float();
			arr[i][1] = (float) Utils::rand_float();
		}
	}
}

void Array::fftwComplex::fillArray_2D_flat(fftwf_complex* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			if (Seq_Rnd)
			{
				arr[index][0] = (float) index;
				arr[index][1] = (float) index;
			}
			else
			{
				arr[index][0] = (float) Utils::rand_float();
				arr[index][1] = (float) Utils::rand_float();
			}
		}
}

void Array::fftwComplex::fillArray_3D_flat(fftwf_complex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				int index = (i + (size_X * j) + (size_X * size_Y * k));

				if (Seq_Rnd)
				{
					arr[index][0] = (float) index;
					arr[index][1] = (float) index;
				}
				else
				{
					arr[index][0] = (float) Utils::rand_float();
					arr[index][1] = (float) Utils::rand_float();
				}
			}
}

void Array::fftwComplex::fillArray_2D(fftwf_complex** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
			{
				arr[i][j][0] = (float) ctr;
				arr[i][j][1] = (float) ctr++;
			}
			else
			{
				arr[i][j][0] = (float) Utils::rand_float();
				arr[i][j][1] = (float) Utils::rand_float();
			}
		}
}

void Array::fftwComplex::fillArray_3D(fftwf_complex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
				{
					arr[i][j][k][0] = (float) ctr;
					arr[i][j][k][1] = (float) ctr++;
				}
				else
				{
					arr[i][j][k][0] = (float) Utils::rand_float();
					arr[i][j][k][1] = (float) Utils::rand_float();
				}
			}
}

void Array::fftwDoubleComplex::fillArray_1D(fftw_complex* arr, int size_X, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
	{
		if (Seq_Rnd)
		{
			arr[i][0] = (double) i;
			arr[i][1] = (double) i;
		}
		else
		{
			arr[i][0] = (double) Utils::rand_double();
			arr[i][1] = (double) Utils::rand_double();
		}
	}
}

void Array::fftwDoubleComplex::fillArray_2D_flat(fftw_complex* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			if (Seq_Rnd)
			{
				arr[index][0] = (double) index;
				arr[index][1] = (double) index;
			}
			else
			{
				arr[index][0] = (double) Utils::rand_double();
				arr[index][1] = (double) Utils::rand_double();
			}
		}
}

void Array::fftwDoubleComplex::fillArray_3D_flat(fftw_complex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				int index = (i + (size_X * j) + (size_X * size_Y * k));

				if (Seq_Rnd)
				{
					arr[index][0] = (double) index;
					arr[index][1] = (double) index;
				}
				else
				{
					arr[index][0] = (double) Utils::rand_double();
					arr[index][1] = (double) Utils::rand_double();
				}
			}
}

void Array::fftwDoubleComplex::fillArray_2D(fftw_complex** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
			{
				arr[i][j][0] = (double) ctr;
				arr[i][j][1] = (double) ctr++;
			}
			else
			{
				arr[i][j][0] = (double) Utils::rand_double();
				arr[i][j][1] = (double) Utils::rand_double();
			}
		}
}

void Array::fftwDoubleComplex::fillArray_3D(fftw_complex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
				{
					arr[i][j][k][0] = (double) ctr;
					arr[i][j][k][1] = (double) ctr++;
				}
				else
				{
					arr[i][j][k][0] = (double) Utils::rand_double();
					arr[i][j][k][1] = (double) Utils::rand_double();
				}
			}
}



