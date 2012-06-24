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


#define START_ROW_DATA 3

namespace iB_cuFFTShift_2D
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	/* @ Host arrays */
	float* arr_2D_flat_float;
	float** arr_2D_float;

	double* arr_2D_flat_double;
	double** arr_2D_double;

	/* @ Device array */
	float* in_dev_arr_2D_flat_float;
	float* out_dev_arr_2D_flat_float;

	double* in_dev_arr_2D_flat_double;
	double* out_dev_arr_2D_flat_double;

	/* @ Profilers */
	cudaProfile* cuProfile;
	durationStruct* cpuProfile;
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
			// Rows
			xlSheet->writeStr(2, ((iLoop * 4 ) + 0), "ns");
			xlSheet->writeStr(3, ((iLoop * 4 ) + 0), "us");
			xlSheet->writeStr(4, ((iLoop * 4 ) + 0), "ms");
			xlSheet->writeStr(5, ((iLoop * 4 ) + 0), "s");
			xlSheet->writeStr(6, ((iLoop * 4 ) + 0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 2D, Flat, Device
			arr_2D_float = MEM_ALLOC_2D_FLOAT(size_X, size_Y);
			arr_2D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(float);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_float), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_float), devMem);

			// Filling arrays: 2D, Flat
			Array::fillArray_2D_float(arr_2D_float, size_X, size_Y, 1);
			Array::fillArray_2D_flat_float(arr_2D_flat_float, size_X, size_Y, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_2D_float[i][j]);
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_float = FFT::FFT_Shift_2D_float(arr_2D_float, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_2D_float[i][j]);
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_float(arr_2D_flat_float, in_dev_arr_2D_flat_float, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D(cuBlock, cuGrid, out_dev_arr_2D_flat_float, in_dev_arr_2D_flat_float, size_X, cuProfile);

			Array::zeroArray_2D_flat_float(arr_2D_flat_float , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_float(arr_2D_flat_float, out_dev_arr_2D_flat_float, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_float));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_float));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_2D_flat_float[ctr]);
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_FLOAT(arr_2D_float, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_float);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}

void iB_cuFFTShift_2D::FFTShift_2D_Double
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Double, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Double Case
	 **********************************************************/

	if (xlSheet)
	{
		for (int iLoop = 0; iLoop < nLoop; iLoop++)
		{
			// Rows
			xlSheet->writeStr(2, ((iLoop * 4 ) + 0), "ns");
			xlSheet->writeStr(3, ((iLoop * 4 ) + 0), "us");
			xlSheet->writeStr(4, ((iLoop * 4 ) + 0), "ms");
			xlSheet->writeStr(5, ((iLoop * 4 ) + 0), "s");
			xlSheet->writeStr(6, ((iLoop * 4 ) + 0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 2D, Flat, Device
			arr_2D_double = MEM_ALLOC_2D_DOUBLE(size_X, size_Y);
			arr_2D_flat_double = MEM_ALLOC_1D_DOUBLE(size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(double);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_double), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_double), devMem);

			// Filling arrays: 2D, Flat
			Array::fillArray_2D_double(arr_2D_double, size_X, size_Y, 1);
			Array::fillArray_2D_flat_double(arr_2D_flat_double, size_X, size_Y, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_2D_double[i][j]);
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_double = FFT::FFT_Shift_2D_double(arr_2D_double, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_2D_double[i][j]);
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_2D_flat_double[ctr]);
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_double(arr_2D_flat_double, in_dev_arr_2D_flat_double, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D_Double(cuBlock, cuGrid, out_dev_arr_2D_flat_double, in_dev_arr_2D_flat_double, size_X, cuProfile);

			Array::zeroArray_2D_flat_double(arr_2D_flat_double , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_double(arr_2D_flat_double, out_dev_arr_2D_flat_double, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_double));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_double));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == size_X * size_Y - 1)
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_2D_flat_double[ctr]);
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_DOUBLE(arr_2D_double, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_double);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}
