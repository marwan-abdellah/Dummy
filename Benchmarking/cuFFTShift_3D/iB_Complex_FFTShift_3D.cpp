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

#include "iB_Complex_FFTShift_3D.h"
#include <stdio.h>
#include <stdlib.h>


#define START_ROW_DATA 3

namespace iB_Complex_FFTShift_3D
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	/* @ Host arrays */
	cufftComplex* arr_3D_flat_cuComplex;
	cufftComplex*** arr_3D_cuComplex;

	cufftDoubleComplex* arr_3D_flat_cuDoubleComplex;
	cufftDoubleComplex*** arr_3D_cuDoubleComplex;

	/* @ Device array */
	cufftComplex* in_dev_arr_3D_flat_cuComplex;
	cufftComplex* out_dev_arr_3D_flat_cuComplex;

	cufftDoubleComplex* in_dev_arr_3D_flat_cuDoubleComplex;
	cufftDoubleComplex* out_dev_arr_3D_flat_cuDoubleComplex;

	/* @ Profilers */
	cudaProfile* cuProfile;
	durationStruct* cpuProfile;

	cudaProfile* cuTotalProfile;
	durationStruct* cpuTotalProfile;
}

void iB_Complex_FFTShift_3D::FFTShift_3D_Float_Seq
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 						\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		// Averaging Profiles
		cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
		cuTotalProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

		// Initializing average profilers
		cpuTotalProfile->unit_NanoSec = 0;
		cpuTotalProfile->unit_MicroSec = 0;
		cpuTotalProfile->unit_MilliSec = 0;
		cpuTotalProfile->unit_Sec = 0;
		cuTotalProfile->kernelDuration = 0;

		// Rows
		xlSheet->writeStr(10, (0), "ns");
		xlSheet->writeStr(11, (0), "us");
		xlSheet->writeStr(12, (0), "ms");
		xlSheet->writeStr(13, (0), "s");
		xlSheet->writeNum(10, 3, nLoop);

		// Headers
		xlSheet->writeStr(9, 1, "CPU Time");
		xlSheet->writeStr(9, 2, "GPU Time");
		xlSheet->writeStr(9, 3, "N ");

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

			// Allocation: 3D, Flat, Device
			arr_3D_cuComplex = MEM_ALLOC_3D_CUFFTCOMPLEX(size_X, size_Y, size_Z);
			arr_3D_flat_cuComplex = MEM_ALLOC_1D(cufftComplex, size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_cuComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_cuComplex), devMem);

			// Filling arrays: 3D, Flat
			Array::cuComplex::fillArray_3D(arr_3D_cuComplex, size_X, size_Y, size_Z, 1);
			Array::cuComplex::fillArray_3D_flat(arr_3D_flat_cuComplex, size_X, size_Y, size_Z, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuComplex[i][j][k].y);
						}
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_cuComplex = FFT::FFT_Shift_3D_cuComplex(arr_3D_cuComplex, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuComplex[i][j][k].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Adding the timing to the average profiler
			cpuTotalProfile->unit_NanoSec += cpuProfile->unit_NanoSec;
			cpuTotalProfile->unit_MicroSec += cpuProfile->unit_MicroSec;
			cpuTotalProfile->unit_MilliSec += cpuProfile->unit_MilliSec;
			cpuTotalProfile->unit_Sec += cpuProfile->unit_Sec;

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuComplex[ctr].y);
						}
						ctr++;
					}


			// Uploading input array
			cuUtils::upload_3D_cuComplex(arr_3D_flat_cuComplex, in_dev_arr_3D_flat_cuComplex, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Complex(cuBlock, cuGrid, out_dev_arr_3D_flat_cuComplex, in_dev_arr_3D_flat_cuComplex, size_X, cuProfile);

			Array::cuComplex::zeroArray_3D_flat(arr_3D_flat_cuComplex , size_X, size_Y, size_Z);


			// Downloading output array
			cuUtils::download_3D_cuComplex(arr_3D_flat_cuComplex, out_dev_arr_3D_flat_cuComplex, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_cuComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_cuComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuComplex[ctr].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_CUFFTCOMPLEX(arr_3D_cuComplex, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_cuComplex);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

		// Priting average profile data
		xlSheet->writeNum(10, 1, cpuTotalProfile->unit_NanoSec / nLoop);
		xlSheet->writeNum(11, 1, cpuTotalProfile->unit_MicroSec / nLoop);
		xlSheet->writeNum(12, 1, cpuTotalProfile->unit_MilliSec / nLoop);
		xlSheet->writeNum(13, 1, cpuTotalProfile->unit_Sec / nLoop);

		xlSheet->writeNum(10, 2, (cuTotalProfile->kernelDuration * 1000 * 1000) / nLoop);
		xlSheet->writeNum(11, 2, (cuTotalProfile->kernelDuration * 1000) / nLoop);
		xlSheet->writeNum(12, 2, (cuTotalProfile->kernelDuration) / nLoop);
		xlSheet->writeNum(13, 2, (cuTotalProfile->kernelDuration / 1000) / nLoop);

		// Releasing the averaging profilers
		FREE_MEM_1D(cuTotalProfile);
		FREE_MEM_1D(cpuTotalProfile);

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}

void iB_Complex_FFTShift_3D::FFTShift_3D_Float_Rnd
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 						\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		// Averaging Profiles
		cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
		cuTotalProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

		// Initializing average profilers
		cpuTotalProfile->unit_NanoSec = 0;
		cpuTotalProfile->unit_MicroSec = 0;
		cpuTotalProfile->unit_MilliSec = 0;
		cpuTotalProfile->unit_Sec = 0;
		cuTotalProfile->kernelDuration = 0;

		// Rows
		xlSheet->writeStr(10, (0), "ns");
		xlSheet->writeStr(11, (0), "us");
		xlSheet->writeStr(12, (0), "ms");
		xlSheet->writeStr(13, (0), "s");
		xlSheet->writeNum(10, 3, nLoop);

		// Headers
		xlSheet->writeStr(9, 1, "CPU Time");
		xlSheet->writeStr(9, 2, "GPU Time");
		xlSheet->writeStr(9, 3, "N ");

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

			// Allocation: 3D, Flat, Device
			arr_3D_cuComplex = MEM_ALLOC_3D_CUFFTCOMPLEX(size_X, size_Y, size_Z);
			arr_3D_flat_cuComplex = MEM_ALLOC_1D(cufftComplex, size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_cuComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_cuComplex), devMem);

			// Filling arrays: 3D, Flat
			Array::cuComplex::fillArray_3D(arr_3D_cuComplex, size_X, size_Y, size_Z, 0);
			Array::cuComplex::fillArray_3D_flat(arr_3D_flat_cuComplex, size_X, size_Y, size_Z, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuComplex[i][j][k].y);
						}
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_cuComplex = FFT::FFT_Shift_3D_cuComplex(arr_3D_cuComplex, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuComplex[i][j][k].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Adding the timing to the average profiler
			cpuTotalProfile->unit_NanoSec += cpuProfile->unit_NanoSec;
			cpuTotalProfile->unit_MicroSec += cpuProfile->unit_MicroSec;
			cpuTotalProfile->unit_MilliSec += cpuProfile->unit_MilliSec;
			cpuTotalProfile->unit_Sec += cpuProfile->unit_Sec;

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuComplex[ctr].y);
						}
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_cuComplex(arr_3D_flat_cuComplex, in_dev_arr_3D_flat_cuComplex, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Complex(cuBlock, cuGrid, out_dev_arr_3D_flat_cuComplex, in_dev_arr_3D_flat_cuComplex, size_X, cuProfile);

			Array::cuComplex::zeroArray_3D_flat(arr_3D_flat_cuComplex , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_cuComplex(arr_3D_flat_cuComplex, out_dev_arr_3D_flat_cuComplex, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_cuComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_cuComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuComplex[ctr].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_CUFFTCOMPLEX(arr_3D_cuComplex, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_cuComplex);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

		// Priting average profile data
		xlSheet->writeNum(10, 1, cpuTotalProfile->unit_NanoSec / nLoop);
		xlSheet->writeNum(11, 1, cpuTotalProfile->unit_MicroSec / nLoop);
		xlSheet->writeNum(12, 1, cpuTotalProfile->unit_MilliSec / nLoop);
		xlSheet->writeNum(13, 1, cpuTotalProfile->unit_Sec / nLoop);

		xlSheet->writeNum(10, 2, (cuTotalProfile->kernelDuration * 1000 * 1000) / nLoop);
		xlSheet->writeNum(11, 2, (cuTotalProfile->kernelDuration * 1000) / nLoop);
		xlSheet->writeNum(12, 2, (cuTotalProfile->kernelDuration) / nLoop);
		xlSheet->writeNum(13, 2, (cuTotalProfile->kernelDuration / 1000) / nLoop);

		// Releasing the averaging profilers
		FREE_MEM_1D(cuTotalProfile);
		FREE_MEM_1D(cpuTotalProfile);

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}


void iB_Complex_FFTShift_3D::FFTShift_3D_Double_Seq
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Complex Float, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Double Case
	 **********************************************************/

	if (xlSheet)
	{
		// Averaging Profiles
		cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
		cuTotalProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

		// Initializing average profilers
		cpuTotalProfile->unit_NanoSec = 0;
		cpuTotalProfile->unit_MicroSec = 0;
		cpuTotalProfile->unit_MilliSec = 0;
		cpuTotalProfile->unit_Sec = 0;
		cuTotalProfile->kernelDuration = 0;

		// Rows
		xlSheet->writeStr(10, (0), "ns");
		xlSheet->writeStr(11, (0), "us");
		xlSheet->writeStr(12, (0), "ms");
		xlSheet->writeStr(13, (0), "s");
		xlSheet->writeNum(10, 3, nLoop);

		// Headers
		xlSheet->writeStr(9, 1, "CPU Time");
		xlSheet->writeStr(9, 2, "GPU Time");
		xlSheet->writeStr(9, 3, "N ");

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

			// Allocation: 3D, Flat, Device
			arr_3D_cuDoubleComplex = MEM_ALLOC_3D_CUFFTDOUBLECOMPLEX(size_X, size_Y, size_Z);
			arr_3D_flat_cuDoubleComplex = MEM_ALLOC_1D(cufftDoubleComplex, size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_cuDoubleComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_cuDoubleComplex), devMem);

			// Filling arrays: 3D, Flat
			Array::cuDoubleComplex::fillArray_3D(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z, 1);
			Array::cuDoubleComplex::fillArray_3D_flat(arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_cuDoubleComplex = FFT::FFT_Shift_3D_cuDoubleComplex(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8) + 4), cpuProfile->unit_Sec);

			// Adding the timing to the average profiler
			cpuTotalProfile->unit_NanoSec += cpuProfile->unit_NanoSec;
			cpuTotalProfile->unit_MicroSec += cpuProfile->unit_MicroSec;
			cpuTotalProfile->unit_MilliSec += cpuProfile->unit_MilliSec;
			cpuTotalProfile->unit_Sec += cpuProfile->unit_Sec;

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_cuDoubleComplex(arr_3D_flat_cuDoubleComplex, in_dev_arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Double_Complex(cuBlock, cuGrid, out_dev_arr_3D_flat_cuDoubleComplex, in_dev_arr_3D_flat_cuDoubleComplex, size_X, cuProfile);

			Array::cuDoubleComplex::zeroArray_3D_flat(arr_3D_flat_cuDoubleComplex , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_cuDoubleComplex(arr_3D_flat_cuDoubleComplex, out_dev_arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_cuDoubleComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_cuDoubleComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_CUFFTDOUBLECOMPLEX(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_cuDoubleComplex);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

		// Priting average profile data
		xlSheet->writeNum(10, 1, cpuTotalProfile->unit_NanoSec / nLoop);
		xlSheet->writeNum(11, 1, cpuTotalProfile->unit_MicroSec / nLoop);
		xlSheet->writeNum(12, 1, cpuTotalProfile->unit_MilliSec / nLoop);
		xlSheet->writeNum(13, 1, cpuTotalProfile->unit_Sec / nLoop);

		xlSheet->writeNum(10, 2, (cuTotalProfile->kernelDuration * 1000 * 1000) / nLoop);
		xlSheet->writeNum(11, 2, (cuTotalProfile->kernelDuration * 1000) / nLoop);
		xlSheet->writeNum(12, 2, (cuTotalProfile->kernelDuration) / nLoop);
		xlSheet->writeNum(13, 2, (cuTotalProfile->kernelDuration / 1000) / nLoop);

		// Releasing the averaging profilers
		FREE_MEM_1D(cuTotalProfile);
		FREE_MEM_1D(cpuTotalProfile);

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}

void iB_Complex_FFTShift_3D::FFTShift_3D_Double_Rnd
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Complex Double, Array:" + ITS(size_X) + "x" + ITS(size_Y)	\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 						\
		+ " Grid:" + ITS(cuGrid.x) + "x" + ITS(cuGrid.y));

	/**********************************************************
	 * Float Case
	 **********************************************************/

	if (xlSheet)
	{
		// Averaging Profiles
		cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
		cuTotalProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

		// Initializing average profilers
		cpuTotalProfile->unit_NanoSec = 0;
		cpuTotalProfile->unit_MicroSec = 0;
		cpuTotalProfile->unit_MilliSec = 0;
		cpuTotalProfile->unit_Sec = 0;
		cuTotalProfile->kernelDuration = 0;

		// Rows
		xlSheet->writeStr(10, (0), "ns");
		xlSheet->writeStr(11, (0), "us");
		xlSheet->writeStr(12, (0), "ms");
		xlSheet->writeStr(13, (0), "s");
		xlSheet->writeNum(10, 3, nLoop);

		// Headers
		xlSheet->writeStr(9, 1, "CPU Time");
		xlSheet->writeStr(9, 2, "GPU Time");
		xlSheet->writeStr(9, 3, "N ");

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

			// Allocation: 3D, Flat, Device
			arr_3D_cuDoubleComplex = MEM_ALLOC_3D_CUFFTDOUBLECOMPLEX(size_X, size_Y, size_Z);
			arr_3D_flat_cuDoubleComplex = MEM_ALLOC_1D(cufftDoubleComplex, size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_cuDoubleComplex), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_cuDoubleComplex), devMem);

			// Filling arrays: 3D, Flat
			Array::cuDoubleComplex::fillArray_3D(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z, 0);
			Array::cuDoubleComplex::fillArray_3D_flat(arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 1), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8) + 2), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_cuDoubleComplex = FFT::FFT_Shift_3D_cuDoubleComplex(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 3), arr_3D_cuDoubleComplex[i][j][k].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 4), arr_3D_cuDoubleComplex[i][j][k].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 4), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 4), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 4), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 8 ) + 4), cpuProfile->unit_Sec);

			// Adding the timing to the average profiler
			cpuTotalProfile->unit_NanoSec += cpuProfile->unit_NanoSec;
			cpuTotalProfile->unit_MicroSec += cpuProfile->unit_MicroSec;
			cpuTotalProfile->unit_MilliSec += cpuProfile->unit_MilliSec;
			cpuTotalProfile->unit_Sec += cpuProfile->unit_Sec;

			// Printing GPU input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 5), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 6), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_cuDoubleComplex(arr_3D_flat_cuDoubleComplex, in_dev_arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Double_Complex(cuBlock, cuGrid, out_dev_arr_3D_flat_cuDoubleComplex, in_dev_arr_3D_flat_cuDoubleComplex, size_X, cuProfile);

			Array::cuDoubleComplex::zeroArray_3D_flat(arr_3D_flat_cuDoubleComplex , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_cuDoubleComplex(arr_3D_flat_cuDoubleComplex, out_dev_arr_3D_flat_cuDoubleComplex, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_cuDoubleComplex));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_cuDoubleComplex));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
						{
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						if(ctr == size_X * size_Y * size_Z - 1)
						{
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 7), arr_3D_flat_cuDoubleComplex[ctr].x);
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 8 ) + 8), arr_3D_flat_cuDoubleComplex[ctr].y);
						}
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 8 ) + 8), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 8 ) + 8), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 8 ) + 8), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 8 ) + 8), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_CUFFTDOUBLECOMPLEX(arr_3D_cuDoubleComplex, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_cuDoubleComplex);
			FREE_MEM_1D(cuProfile);
			FREE_MEM_1D(cpuProfile);
		}

		// Priting average profile data
		xlSheet->writeNum(10, 1, cpuTotalProfile->unit_NanoSec / nLoop);
		xlSheet->writeNum(11, 1, cpuTotalProfile->unit_MicroSec / nLoop);
		xlSheet->writeNum(12, 1, cpuTotalProfile->unit_MilliSec / nLoop);
		xlSheet->writeNum(13, 1, cpuTotalProfile->unit_Sec / nLoop);

		xlSheet->writeNum(10, 2, (cuTotalProfile->kernelDuration * 1000 * 1000) / nLoop);
		xlSheet->writeNum(11, 2, (cuTotalProfile->kernelDuration * 1000) / nLoop);
		xlSheet->writeNum(12, 2, (cuTotalProfile->kernelDuration) / nLoop);
		xlSheet->writeNum(13, 2, (cuTotalProfile->kernelDuration / 1000) / nLoop);

		// Releasing the averaging profilers
		FREE_MEM_1D(cuTotalProfile);
		FREE_MEM_1D(cpuTotalProfile);

	}
	else
	{
		INFO("No valid xlSheet was created, EXITTING ...");
		EXIT(0);
	}
}

