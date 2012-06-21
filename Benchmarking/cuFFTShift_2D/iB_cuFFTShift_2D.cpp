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

#include "iB_cuFFTShift_2D.h"
#include <stdio.h>
#include <stdlib.h>


namespace iB_cuFFTShift_2D
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	/* @ Host arrays */
	float* arr_2D_flat_float;
	float** arr_2D_float;

	/* @ Device array */
	float* dev_arr_2D_flat_float;
}

void iB_cuFFTShift_2D::FFTShift_2D_Float
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

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
					{}//xlSheet->writeNum((ctr++) + 2, iLoop * 4, arr_2D_float[i][j]);

			// FFT shift operation - CPU
			arr_2D_float = FFT::FFT_Shift_2D_float(arr_2D_float, size_X, size_Y);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					{}//xlSheet->writeNum((ctr++) + 2, ((iLoop * 4 ) + 1), arr_2D_float[i][j]);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					//xlSheet->writeNum(ctr + 2, ((iLoop * 4 ) + 2), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Uploading array
			cuUtils::upload_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// CUDA Gridding
			//dim3 cuBlock(512, 512, 1);
			//dim3 cuGrid(size_X / cuBlock.x, size_Y/ cuBlock.y, 1);

			// FFT shift
			cuFFTShift_2D(cuBlock, cuGrid, dev_arr_2D_flat_float, dev_arr_2D_flat_float, size_X);

			// Downloading array
			cuUtils::download_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*)dev_arr_2D_flat_float));


			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					//xlSheet->writeNum((ctr) + 2, ((iLoop * 4 ) + 3), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Dellocating memory
			FREE_MEM_2D_FLOAT(arr_2D_float, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_float);
		}

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}

void iB_cuFFTShift_2D::FFTShift_2D_Float_CUDA
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		for (int iLoop = 0; iLoop < nLoop; iLoop++)
		{
			// Headers
			xlSheet->writeStr(1, ((iLoop * 2) + 0), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 2) + 1), "O-GPU");

			// Allocation: Flat, Device
			arr_2D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(float);
			cudaMalloc((void**)(&dev_arr_2D_flat_float), devMem);

			// Filling arrays: Flat
			Array::fillArray_2D_flat_float(arr_2D_flat_float, size_X, size_Y, 1);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					//xlSheet->writeNum(ctr + 2, ((iLoop * 2 ) + 0), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Uploading array
			cuUtils::upload_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// FFT shift
			cuFFTShift_2D(cuBlock, cuGrid, dev_arr_2D_flat_float, dev_arr_2D_flat_float, size_X);

			// Downloading array
			cuUtils::download_2D_float(arr_2D_flat_float, dev_arr_2D_flat_float, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*)dev_arr_2D_flat_float));


			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					//xlSheet->writeNum((ctr) + 2, ((iLoop * 2 ) + 1), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Dellocating memory
			FREE_MEM_1D(arr_2D_flat_float);
		}

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}
