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

#include "iB_FFTShift.h"
#include <stdio.h>
#include <stdlib.h>


namespace iB_FFTShift
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	float* arr_1D_flat_float;
	float* arr_2D_flat_float;
	float* arr_3D_flat_float;
	float** arr_2D_float;
	float*** arr_3D_float;

	float* dev_arr_1D_flat_float;
	float* dev_arr_2D_flat_float;

	float* dev_arr_3D_flat_float_input;
	float* dev_arr_3D_flat_float_output;
}

void iB_FFTShift::FFTShift_2D_Float(int size_X, int size_Y, Sheet* xlSheet, int nLoop)
{
	LOG();

	INFO("2D FFT Shift Float - CPU " + ITS(size_X) + "x" + ITS(size_Y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		for (int iLoop = 0; iLoop < nLoop; iLoop++)
		{
			// Headers
			xlSheet->writeStr(1, ((iLoop * 4) + 0), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "O-GPU");

			// Allocation: 2D, Flat, Device
			arr_2D_float = MEM_ALLOC_2D_FLOAT(size_X, size_Y);
			arr_2D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(float);
			cudaMalloc((void**)(&dev_arr_2D_flat_float), devMem);

			// Filling arrays: 2D, Flat
			Array::fillArray_2D_float(arr_2D_float, size_X, size_Y, 1);
			Array::fillArray_2D_flat_float(arr_2D_flat_float, size_X, size_Y, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, iLoop * 4, arr_2D_float[i][j]);

			// FFT shift operation - CPU
			arr_2D_float = FFT::FFT_Shift_2D_float(arr_2D_float, size_X, size_Y);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, ((iLoop * 4 ) + 1), arr_2D_float[i][j]);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					xlSheet->writeNum(ctr + 2, ((iLoop * 4 ) + 2), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Uploading array
			cuUtils::upload_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// CUDA Gridding
			dim3 cuBlock(512, 512, 1);
			dim3 cuGrid(size_X / cuBlock.x, size_Y/ cuBlock.y, 1);

			// FFT shift
			cuFFTShift_2D( cuBlock, cuGrid, dev_arr_2D_flat_float, dev_arr_2D_flat_float, size_X);

			// Downloading array
			cuUtils::download_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					xlSheet->writeNum((ctr) + 2, ((iLoop * 4 ) + 3), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Dellocating memory
			FREE_MEM_2D_FLOAT(arr_2D_float, size_X, size_Y);
		}

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}























void iB_FFTShift::FFTShift_2D_CUDA(int size_X, int size_Y)
{
	LOG();

	INFO("2D FFT Shift - CUDA");

	// Host allocation
	arr_2D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y);

	// Filling array
	Array::fillArray_2D_flat_float(arr_2D_flat_float, size_X, size_Y, 1);

	// Printing input
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			printf("Input \t %f \n", arr_2D_flat_float[ctr++]);

	// Device allocation
	int devMem = size_X * size_Y * sizeof(float);
	cudaMalloc((void**)(&dev_arr_2D_flat_float), devMem);

	// Uploading array
	cuUtils::upload_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

	// CUDA Gridding
	dim3 cuBlock(4, 4, 1);
	dim3 cuGrid(size_X / cuBlock.x, size_Y/ cuBlock.y, 1);

	// FFT shift
	cuFFTShift_2D( cuBlock, cuGrid, dev_arr_2D_flat_float, dev_arr_2D_flat_float, size_X);

	// Downloading array
	cuUtils::download_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

	// Printing output
	ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			printf("output \t %f \n", arr_2D_flat_float[ctr++]);
}


void iB_FFTShift::FFTShift_3D_CPU(int size_X, int size_Y, int size_Z)
{
	LOG();

	INFO("3D FFT Shift - CPU");

	// Allocation
	arr_3D_float = MEM_ALLOC_3D_FLOAT(size_X, size_Y, size_Z);

	// Filling array
	Array::fillArray_3D_float(arr_3D_float, size_X, size_Y, size_Z, 1);

	// Printing input
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				printf("Input \t %f \n", arr_3D_float[i][j][k]);
	// 3D FFT shift
	arr_3D_float = FFT::FFT_Shift_3D_float(arr_3D_float, size_X, size_Y, size_Z);

	// Printing output
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				printf("Output \t %f \n", arr_3D_float[i][j][k]);

}

void iB_FFTShift::FFTShift_3D_CUDA(int size_X, int size_Y, int size_Z)
{
	LOG();

	INFO("3D FFT Shift - CUDA");

	// Host allocation
	arr_3D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y * size_Z);

	// Filling array
	Array::fillArray_3D_flat_float(arr_3D_flat_float, size_X, size_Y, size_Z, 1);

	// Printing input
	int ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				printf("Input \t %f \n", arr_3D_flat_float[ctr++]);

	// Device allocation
	// Note: We must use different arrays for the input and the output
	// 		 for this implementation. "see core implementation"
	int devMem = size_X * size_Y * size_Z * sizeof(float);
	cudaMalloc((void**)(&dev_arr_3D_flat_float_input), devMem);
	cudaMalloc((void**)(&dev_arr_3D_flat_float_output), devMem);

	// Uploading array
	cuUtils::upload_3D_float(arr_3D_flat_float, dev_arr_3D_flat_float_input, size_X, size_Y, size_Z);

	// CUDA Gridding
	dim3 cuBlock(4, 4, 1);
	dim3 cuGrid(size_X / cuBlock.x, size_Y/ cuBlock.y, 1);

	float* dev_arr_3D_flat_float_temp;
	cudaMalloc((void**)(&dev_arr_3D_flat_float_temp), devMem);

	// FFT shift
	cuFFTShift_3D( cuBlock, cuGrid, dev_arr_3D_flat_float_output, dev_arr_3D_flat_float_input, size_X);

	// Downloading array
	cuUtils::download_3D_float(arr_3D_flat_float, dev_arr_3D_flat_float_output, size_X, size_Y, size_Z);

	// Printing output
	ctr = 0;
	for (int i = 0; i < size_X; i++)
		for (int j = 0; j < size_Y; j++)
			for (int k = 0; k < size_Z; k++)
				printf("Output \t %f \n", arr_3D_flat_float[ctr++]);
}


