/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: FFTShift.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "FFTShift.h"

float* FFT_Shift_1D_float(float* input, int nX)
{
	LOG();

	const int N = nX;

	float* output;
	output = MEM_ALLOC_1D(float, N);

	for(int i = 0; i < N/2; i++)
	{
		output[(N/2) + i] = input[i];
		output[i] = input[(N/2) + i];
	}

	return output;
}

float** FFT::FFT_Shift_2D_float(float** input, int nX, int nY)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		float** output;
		output = MEM_ALLOC_2D_FLOAT(N, N);

		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				output[(N/2) + i][(N/2) + j] = input[i][j];
				output[i][j] = input[(N/2) + i][(N/2) + j];

				output[i][(N/2) + j] = input[(N/2) + i][j];
				output[(N/2) + i][j] = input[i][(N/2) + j];
			}

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}

	return NULL;
}

float*** FFT::FFT_Shift_3D_float(float*** input, int nX, int nY, int nZ)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		float ***output;;
		output = MEM_ALLOC_3D_FLOAT(N, N, N);

		/* Doing the 3D FFT shift operation */
		for (int k = 0; k < N/2; k++)
			for (int i = 0; i < N/2; i++)
				for(int j = 0; j < N/2; j++)
				{
					output[(N/2) + i][(N/2) + j][(N/2) + k] = input[i][j][k];
					output[i][j][k] = input[(N/2) + i][(N/2) + j][(N/2) + k];

					output[(N/2) + i][j][(N/2) + k] = input[i][(N/2) + j][k];
					output[i][(N/2) + j][k] = input[(N/2) + i][j][(N/2) + k];

					output[i][(N/2) + j][(N/2) + k] = input[(N/2) + i][j][k];
					output[(N/2) + i][j][k] = input[i][(N/2) + j][(N/2) + k];

					output[i][j][(N/2) + k] = input[(N/2) + i][(N/2) + j][k];
					output[(N/2) + i][(N/2) + j][k] = input[i][j][(N/2) + k];
				}

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

float* FFT::repack_2D_float(float** input_2D, int nX, int nY)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		float *output_1D;;
		output_1D = MEM_ALLOC_1D_FLOAT(N  * N);

		int ctr = 0;
		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				output_1D[ctr] = input_2D[i][j];
				ctr++;
			}

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}
	return NULL;
}

float* FFT::repack_3D_float(float*** input_3D, int nX, int nY, int nZ)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		float *output_1D;;
		output_1D = MEM_ALLOC_1D_FLOAT(N * N * N);

		// Re-packing the 3D volume into 1D array
		int ctr = 0;
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
				{
					output_1D[ctr] = input_3D[i][j][k];
					ctr++;
				}

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}
