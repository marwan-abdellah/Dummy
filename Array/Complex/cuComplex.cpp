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
#include "cuComplex.h"

void Array::cuComplex::zeroArray_2D_flat(cufftComplex* arr, int size_X, int size_Y)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			arr[index].x = (cufftReal) 0;
			arr[index].y = (cufftReal) 0;
		}
}

void Array::cuComplex::zeroArray_3D_flat(cufftComplex* arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				arr[ctr].x = 0;
				arr[ctr].y = 0;
				ctr++;
			}
}

void Array::cuComplex::zeroArray_2D(cufftComplex** arr, int size_X, int size_Y)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			arr[i][j].x = (cufftReal) 0;
			arr[i][j].y = (cufftReal) 0;
		}

}

void Array::cuComplex::zeroArray_3D(cufftComplex*** arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				arr[i][j][k].x = (cufftReal) 0;
				arr[i][j][k].y = (cufftReal) 0;
			}
}

void Array::cuDoubleComplex::zeroArray_2D_flat(cufftDoubleComplex* arr, int size_X, int size_Y)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			arr[index].x = (cufftDoubleReal) 0;
			arr[index].y = (cufftDoubleReal) 0;
		}
}

void Array::cuDoubleComplex::zeroArray_3D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				arr[ctr].x = (cufftDoubleReal) 0;
				arr[ctr].y = (cufftDoubleReal) 0;
				ctr++;
			}
}

void Array::cuDoubleComplex::zeroArray_2D(cufftDoubleComplex** arr, int size_X, int size_Y)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			arr[i][j].x = (cufftDoubleReal) 0;
			arr[i][j].y = (cufftDoubleReal) 0;
		}

}

void Array::cuDoubleComplex::zeroArray_3D(cufftDoubleComplex*** arr, int size_X, int size_Y, int size_Z)
{
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				arr[i][j][k].x = (cufftDoubleReal) 0;
				arr[i][j][k].y = (cufftDoubleReal) 0;
			}
}

void Array::cuComplex::fillArray_1D(cufftComplex* arr, int size_X, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
	{
		if (Seq_Rnd)
		{
			arr[i].x = (cufftReal) i;
			arr[i].y = (cufftReal) i;
		}
		else
		{
			arr[i].x = (cufftReal) Utils::rand_float();
			arr[i].y = (cufftReal) Utils::rand_float();
		}
	}
}

void Array::cuComplex::fillArray_2D_flat(cufftComplex* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			if (Seq_Rnd)
			{
				arr[index].x = (cufftReal) index;
				arr[index].y = (cufftReal) index;
			}
			else
			{
				arr[index].x = (cufftReal) Utils::rand_float();
				arr[index].y = (cufftReal) Utils::rand_float();
			}
		}
}

void Array::cuComplex::fillArray_3D_flat(cufftComplex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				int index = (i + (size_X * j) + (size_X * size_Y * k));

				if (Seq_Rnd)
				{
					arr[index].x = (cufftReal) index;
					arr[index].y = (cufftReal) index;
				}
				else
				{
					arr[index].x = (cufftReal) Utils::rand_float();
					arr[index].y = (cufftReal) Utils::rand_float();
				}
			}
}

void Array::cuComplex::fillArray_2D(cufftComplex** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
			{
				arr[i][j].x = (cufftReal) ctr;
				arr[i][j].y = (cufftReal) ctr;
			}
			else
			{
				arr[i][j].x = (cufftReal) Utils::rand_float();
				arr[i][j].y = (cufftReal) Utils::rand_float();
			}

			ctr++;
		}
}

void Array::cuComplex::fillArray_3D(cufftComplex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
				{
					arr[i][j][k].x = (cufftReal) ctr;
					arr[i][j][k].y = (cufftReal) ctr;
				}
				else
				{
					arr[i][j][k].x = (cufftReal) Utils::rand_float();
					arr[i][j][k].y = (cufftReal) Utils::rand_float();
				}

				ctr++;
			}
}

void Array::cuDoubleComplex::fillArray_1D(cufftDoubleComplex* arr, int size_X, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
	{
		if (Seq_Rnd)
		{
			arr[i].x = (cufftDoubleReal) i;
			arr[i].y = (cufftDoubleReal) i;
		}
		else
		{
			arr[i].x = (cufftDoubleReal) Utils::rand_double();
			arr[i].y = (cufftDoubleReal) Utils::rand_double();
		}
	}
}

void Array::cuDoubleComplex::fillArray_2D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			int index = (i + (size_X * j));
			if (Seq_Rnd)
			{
				arr[index].x = (cufftDoubleReal) index;
				arr[index].y = (cufftDoubleReal) index;
			}
			else
			{
				arr[index].x = (cufftDoubleReal) Utils::rand_double();
				arr[index].y = (cufftDoubleReal) Utils::rand_double();
			}
		}
}

void Array::cuDoubleComplex::fillArray_3D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				int index = (i + (size_X * j) + (size_X * size_Y * k));

				if (Seq_Rnd)
				{
					arr[index].x = (cufftDoubleReal) index;
					arr[index].y = (cufftDoubleReal) index;
				}
				else
				{
					arr[index].x = (cufftDoubleReal) Utils::rand_double();
					arr[index].y = (cufftDoubleReal) Utils::rand_double();
				}
			}
}

void Array::cuDoubleComplex::fillArray_2D(cufftDoubleComplex** arr, int size_X, int size_Y, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
		{
			if (Seq_Rnd)
			{
				arr[i][j].x = (cufftDoubleReal) ctr++;
				arr[i][j].y = (cufftDoubleReal) ctr++;
			}
			else
			{
				arr[i][j].x = (cufftDoubleReal) Utils::rand_double();
				arr[i][j].y = (cufftDoubleReal) Utils::rand_double();
			}
		}
}

void Array::cuDoubleComplex::fillArray_3D(cufftDoubleComplex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd)
{
	int ctr = 0;

	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
			{
				if (Seq_Rnd)
				{
					arr[i][j][k].x = (cufftDoubleReal) ctr++;
					arr[i][j][k].y = (cufftDoubleReal) ctr++;
				}
				else
				{
					arr[i][j][k].x = (cufftDoubleReal) Utils::rand_double();
					arr[i][j][k].y = (cufftDoubleReal) Utils::rand_double();
				}
			}
}
