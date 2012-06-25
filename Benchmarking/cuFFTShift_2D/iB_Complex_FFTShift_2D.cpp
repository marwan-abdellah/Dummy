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

#include "iB_Complex_FFTShift_2D.h"
#include <stdio.h>
#include <stdlib.h>


#define START_ROW_DATA 3

namespace iB_Complex_FFTShift_2D
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	/* @ Host arrays */
	cufftComplex* arr_2D_flat_cuComplex;
	cufftComplex** arr_2D_cuComplex;

	cufftDoubleComplex* arr_2D_flat_cuDoubleComplex;
	cufftDoubleComplex** arr_2D_cuDoubleComplex;

	/* @ Device array */
	cufftComplex* in_dev_arr_2D_flat_cuComplex;
	cufftComplex* out_dev_arr_2D_flat_cuComplex;

	cufftDoubleComplex* in_dev_arr_2D_flat_cuDoubleComplex;
	cufftDoubleComplex* out_dev_arr_2D_flat_cuDoubleComplex;

	/* @ Profilers */
	cudaProfile* cuProfile;
	durationStruct* cpuProfile;
}

void iB_Complex_FFTShift_2D::FFTShift_2D_Float_Seq
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 						\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		for (int iLoop = 0; iLoop < nLoop; iLoop++)
		{
			// Rows
			xlSheet->writeStr(2, (0), "ns");
			xlSheet->writeStr(3, (0), "us");
			xlSheet->writeStr(4, (0), "ms");
			xlSheet->writeStr(5, (0), "s");
			xlSheet->writeStr(6, (0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 8) + 1), "I-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 2), "I-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 3), "O-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 4), "O-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 5), "I-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 6), "I-GPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 7), "O-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 8), "O-GPU.y");

			// Allocation: 2D, Flat, Device
			arr_2D_cuComplex = MEM_ALLOC_2D_CUFFTCOMPLEX(size_X, size_Y);
			arr_2D_flat_cuComplex = MEM_ALLOC_1D(cufftComplex, size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(cufftComplex);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_cuComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_cuComplex), devMem);

			// Filling arrays: 2D, Flat
			Array::cuComplex::fillArray_2D(arr_2D_cuComplex, size_X, size_Y, 1);
			Array::cuComplex::fillArray_2D_flat(arr_2D_flat_cuComplex, size_X, size_Y, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuComplex[i][j].y);
					}
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_cuComplex = FFT::FFT_Shift_2D_cuComplex(arr_2D_cuComplex, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuComplex[i][j].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuComplex[ctr].y);
					}
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_cuComplex(arr_2D_flat_cuComplex, in_dev_arr_2D_flat_cuComplex, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D_Complex(cuBlock, cuGrid, out_dev_arr_2D_flat_cuComplex, in_dev_arr_2D_flat_cuComplex, size_X, cuProfile);

			Array::cuComplex::zeroArray_2D_flat(arr_2D_flat_cuComplex , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_cuComplex(arr_2D_flat_cuComplex, out_dev_arr_2D_flat_cuComplex, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_cuComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_cuComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuComplex[ctr].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_CUFFTCOMPLEX(arr_2D_cuComplex, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_cuComplex);
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

void iB_Complex_FFTShift_2D::FFTShift_2D_Float_Rnd
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 						\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		for (int iLoop = 0; iLoop < nLoop; iLoop++)
		{
			// Rows
			xlSheet->writeStr(2, (0), "ns");
			xlSheet->writeStr(3, (0), "us");
			xlSheet->writeStr(4, (0), "ms");
			xlSheet->writeStr(5, (0), "s");
			xlSheet->writeStr(6, (0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 8) + 1), "I-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 2), "I-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 3), "O-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 4), "O-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 5), "I-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 6), "I-GPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 7), "O-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 8), "O-GPU.y");

			// Allocation: 2D, Flat, Device
			arr_2D_cuComplex = MEM_ALLOC_2D_CUFFTCOMPLEX(size_X, size_Y);
			arr_2D_flat_cuComplex = MEM_ALLOC_1D(cufftComplex, size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(cufftComplex);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_cuComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_cuComplex), devMem);

			// Filling arrays: 2D, Flat
			Array::cuComplex::fillArray_2D(arr_2D_cuComplex, size_X, size_Y, 0);
			Array::cuComplex::fillArray_2D_flat(arr_2D_flat_cuComplex, size_X, size_Y, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuComplex[i][j].y);
					}
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_cuComplex = FFT::FFT_Shift_2D_cuComplex(arr_2D_cuComplex, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuComplex[i][j].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuComplex[ctr].y);
					}
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_cuComplex(arr_2D_flat_cuComplex, in_dev_arr_2D_flat_cuComplex, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D_Complex(cuBlock, cuGrid, out_dev_arr_2D_flat_cuComplex, in_dev_arr_2D_flat_cuComplex, size_X, cuProfile);

			Array::cuComplex::zeroArray_2D_flat(arr_2D_flat_cuComplex , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_cuComplex(arr_2D_flat_cuComplex, out_dev_arr_2D_flat_cuComplex, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_cuComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_cuComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuComplex[ctr].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_CUFFTCOMPLEX(arr_2D_cuComplex, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_cuComplex);
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


void iB_Complex_FFTShift_2D::FFTShift_2D_Double_Seq
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
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
			xlSheet->writeStr(2, (0), "ns");
			xlSheet->writeStr(3, (0), "us");
			xlSheet->writeStr(4, (0), "ms");
			xlSheet->writeStr(5, (0), "s");
			xlSheet->writeStr(6, (0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 8) + 1), "I-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 2), "I-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 3), "O-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 4), "O-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 5), "I-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 6), "I-GPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 7), "O-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 8), "O-GPU.y");

			// Allocation: 2D, Flat, Device
			arr_2D_cuDoubleComplex = MEM_ALLOC_2D_CUFFTDOUBLECOMPLEX(size_X, size_Y);
			arr_2D_flat_cuDoubleComplex = MEM_ALLOC_1D(cufftDoubleComplex, size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_cuDoubleComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_cuDoubleComplex), devMem);

			// Filling arrays: 2D, Flat
			Array::cuDoubleComplex::fillArray_2D(arr_2D_cuDoubleComplex, size_X, size_Y, 1);
			Array::cuDoubleComplex::fillArray_2D_flat(arr_2D_flat_cuDoubleComplex, size_X, size_Y, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuDoubleComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuDoubleComplex[i][j].y);
					}
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_cuDoubleComplex = FFT::FFT_Shift_2D_cuDoubleComplex(arr_2D_cuDoubleComplex, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuDoubleComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuDoubleComplex[i][j].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8) + 4), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_cuDoubleComplex(arr_2D_flat_cuDoubleComplex, in_dev_arr_2D_flat_cuDoubleComplex, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D_Double_Complex(cuBlock, cuGrid, out_dev_arr_2D_flat_cuDoubleComplex, in_dev_arr_2D_flat_cuDoubleComplex, size_X, cuProfile);

			Array::cuDoubleComplex::zeroArray_2D_flat(arr_2D_flat_cuDoubleComplex , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_cuDoubleComplex(arr_2D_flat_cuDoubleComplex, out_dev_arr_2D_flat_cuDoubleComplex, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_cuDoubleComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_cuDoubleComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_CUFFTDOUBLECOMPLEX(arr_2D_cuDoubleComplex, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_cuDoubleComplex);
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

void iB_Complex_FFTShift_2D::FFTShift_2D_Double_Rnd
(int size_X, int size_Y, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "2D FFT Shift Complex Double, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
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
			xlSheet->writeStr(2, (0), "ns");
			xlSheet->writeStr(3, (0), "us");
			xlSheet->writeStr(4, (0), "ms");
			xlSheet->writeStr(5, (0), "s");
			xlSheet->writeStr(6, (0), "Errors");

			// Headers
			xlSheet->writeStr(1, ((iLoop * 8) + 1), "I-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 2), "I-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 3), "O-CPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 4), "O-CPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 5), "I-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 6), "I-GPU.y");
			xlSheet->writeStr(1, ((iLoop * 8) + 7), "O-GPU.x");
			xlSheet->writeStr(1, ((iLoop * 8) + 8), "O-GPU.y");

			// Allocation: 2D, Flat, Device
			arr_2D_cuDoubleComplex = MEM_ALLOC_2D_CUFFTDOUBLECOMPLEX(size_X, size_Y);
			arr_2D_flat_cuDoubleComplex = MEM_ALLOC_1D(cufftDoubleComplex, size_X * size_Y);
			int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);
			cudaMalloc((void**)(&in_dev_arr_2D_flat_cuDoubleComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_2D_flat_cuDoubleComplex), devMem);

			// Filling arrays: 2D, Flat
			Array::cuDoubleComplex::fillArray_2D(arr_2D_cuDoubleComplex, size_X, size_Y, 0);
			Array::cuDoubleComplex::fillArray_2D_flat(arr_2D_flat_cuDoubleComplex, size_X, size_Y, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// First & last items only to save writing time to the xlSheet
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuDoubleComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_2D_cuDoubleComplex[i][j].y);
					}
					ctr++;
				}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_2D_cuDoubleComplex = FFT::FFT_Shift_2D_cuDoubleComplex(arr_2D_cuDoubleComplex, size_X, size_Y, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuDoubleComplex[i][j].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_2D_cuDoubleComplex[i][j].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_2D_cuDoubleComplex[i][j].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					ctr++;
				}

			// Uploading input array
			cuUtils::upload_2D_cuDoubleComplex(arr_2D_flat_cuDoubleComplex, in_dev_arr_2D_flat_cuDoubleComplex, size_X, size_Y);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_2D_Double_Complex(cuBlock, cuGrid, out_dev_arr_2D_flat_cuDoubleComplex, in_dev_arr_2D_flat_cuDoubleComplex, size_X, cuProfile);

			Array::cuDoubleComplex::zeroArray_2D_flat(arr_2D_flat_cuDoubleComplex , size_X, size_Y);

			// Downloading output array
			cuUtils::download_2D_cuDoubleComplex(arr_2D_flat_cuDoubleComplex, out_dev_arr_2D_flat_cuDoubleComplex, size_X, size_Y);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_2D_flat_cuDoubleComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_2D_flat_cuDoubleComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					if(ctr == 0)
					{
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					if(ctr == size_X * size_Y - 1)
					{
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_2D_flat_cuDoubleComplex[ctr].x);
						xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_2D_flat_cuDoubleComplex[ctr].y);
					}
					ctr++;
				}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Dellocating: Host memory, profiles
			FREE_MEM_2D_CUFFTDOUBLECOMPLEX(arr_2D_cuDoubleComplex, size_X, size_Y);
			FREE_MEM_1D(arr_2D_flat_cuDoubleComplex);
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

