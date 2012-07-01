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

#include "iB_Real_FFTShift_3D.h"
#include <stdio.h>
#include <stdlib.h>


#define START_ROW_DATA 3

namespace iB_Real_FFTShift_3D
{
	Book* xlBook;
	Sheet* xlSheet;
	int ctr = 0;

	/* @ Host arrays */
	float* arr_3D_flat_float;
	float*** arr_3D_float;

	double* arr_3D_flat_double;
	double*** arr_3D_double;

	/* @ Device array */
	float* in_dev_arr_3D_flat_float;
	float* out_dev_arr_3D_flat_float;

	double* in_dev_arr_3D_flat_double;
	double* out_dev_arr_3D_flat_double;

	/* @ Profilers */
	cudaProfile* cuProfile;
	durationStruct* cpuProfile;

	cudaProfile* cuTotalProfile;
	durationStruct* cpuTotalProfile;
}

void iB_Real_FFTShift_3D::FFTShift_3D_Float_Seq
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Float, Array:" 										\
		+ ITS(size_X) + "x" + ITS(size_Y) + "x" + ITS(size_Z)				\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
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
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 3D, Flat, Device
			arr_3D_float = MEM_ALLOC_3D_FLOAT(size_X, size_Y, size_Z);
			arr_3D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(float);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_float), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_float), devMem);

			// Filling arrays: 3D, Flat
			Array::fillArray_3D_float(arr_3D_float, size_X, size_Y, size_Z, 1);
			Array::fillArray_3D_flat_float(arr_3D_flat_float, size_X, size_Y, size_Z, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_float[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_float[i][j][k]);
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_float = FFT::FFT_Shift_3D_float(arr_3D_float, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_float[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_float[i][j][k]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

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
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_float[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_float[ctr]);
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_float(arr_3D_flat_float, in_dev_arr_3D_flat_float, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D(cuBlock, cuGrid, out_dev_arr_3D_flat_float, in_dev_arr_3D_flat_float, size_X, cuProfile);

			// Clear the array to make sure that the results are true
			Array::zeroArray_3D_flat_float(arr_3D_flat_float , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_float(arr_3D_flat_float, out_dev_arr_3D_flat_float, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_float));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_float));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_float[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_float[ctr]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_FLOAT(arr_3D_float, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_float);
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

void iB_Real_FFTShift_3D::FFTShift_3D_Float_Rnd
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Float, Array:" 										\
		+ ITS(size_X) + "x" + ITS(size_Y) + "x" + ITS(size_Z)				\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
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
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 3D, Flat, Device
			arr_3D_float = MEM_ALLOC_3D_FLOAT(size_X, size_Y, size_Z);
			arr_3D_flat_float = MEM_ALLOC_1D_FLOAT(size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(float);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_float), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_float), devMem);

			// Filling arrays: 3D, Flat
			Array::fillArray_3D_float(arr_3D_float, size_X, size_Y, size_Z, 0);
			Array::fillArray_3D_flat_float(arr_3D_flat_float, size_X, size_Y, size_Z, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_float[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_float[i][j][k]);
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_float = FFT::FFT_Shift_3D_float(arr_3D_float, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_float[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_float[i][j][k]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

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
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_float[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_float[ctr]);
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_float(arr_3D_flat_float, in_dev_arr_3D_flat_float, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D(cuBlock, cuGrid, out_dev_arr_3D_flat_float, in_dev_arr_3D_flat_float, size_X, cuProfile);

			// Clear the array to make sure that the results are true
			Array::zeroArray_3D_flat_float(arr_3D_flat_float , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_float(arr_3D_flat_float, out_dev_arr_3D_flat_float, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_float));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_float));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_float[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_float[ctr]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_FLOAT(arr_3D_float, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_float);
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

void iB_Real_FFTShift_3D::FFTShift_3D_Double_Seq
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Float, Array:" 										\
		+ ITS(size_X) + "x" + ITS(size_Y) + "x" + ITS(size_Z)				\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 				\
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
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 3D, Flat, Device
			arr_3D_double = MEM_ALLOC_3D_DOUBLE(size_X, size_Y, size_Z);
			arr_3D_flat_double = MEM_ALLOC_1D_DOUBLE(size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(double);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_double), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_double), devMem);

			// Filling arrays: 3D, Flat
			Array::fillArray_3D_double(arr_3D_double, size_X, size_Y, size_Z, 1);
			Array::fillArray_3D_flat_double(arr_3D_flat_double, size_X, size_Y, size_Z, 1);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_double[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_double[i][j][k]);
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_double = FFT::FFT_Shift_3D_double(arr_3D_double, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_double[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_double[i][j][k]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

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
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_double[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_double[ctr]);
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_double(arr_3D_flat_double, in_dev_arr_3D_flat_double, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Double(cuBlock, cuGrid, out_dev_arr_3D_flat_double, in_dev_arr_3D_flat_double, size_X, cuProfile);

			// Clear the array to make sure that the results are true
			Array::zeroArray_3D_flat_double(arr_3D_flat_double , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_double(arr_3D_flat_double, out_dev_arr_3D_flat_double, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_double));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_double));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_double[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_double[ctr]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_DOUBLE(arr_3D_double, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_double);
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

void iB_Real_FFTShift_3D::FFTShift_3D_Double_Rnd
(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock)
{
	INFO( "3D FFT Shift Double, Array:" 										\
		+ ITS(size_X) + "x" + ITS(size_Y) + "x" + ITS(size_Z)					\
		+ " Block:" + ITS(cuBlock.x) + "x" + ITS(cuBlock.y) 					\
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
			xlSheet->writeStr(1, ((iLoop * 4) + 1), "I-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 2), "O-CPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 3), "I-GPU");
			xlSheet->writeStr(1, ((iLoop * 4) + 4), "O-GPU");

			// Allocation: 3D, Flat, Device
			arr_3D_double = MEM_ALLOC_3D_DOUBLE(size_X, size_Y, size_Z);
			arr_3D_flat_double = MEM_ALLOC_1D_DOUBLE(size_X * size_Y * size_Z);
			int devMem = size_X * size_Y * size_Z * sizeof(double);
			cudaMalloc((void**)(&in_dev_arr_3D_flat_double), devMem);
			cudaMalloc((void**)(&out_dev_arr_3D_flat_double), devMem);

			// Filling arrays: 3D, Flat
			Array::fillArray_3D_double(arr_3D_double, size_X, size_Y, size_Z, 0);
			Array::fillArray_3D_flat_double(arr_3D_flat_double, size_X, size_Y, size_Z, 0);

			// Printing input
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// First & last items only to save writing time to the xlSheet
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_double[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4) + 1), arr_3D_double[i][j][k]);
						ctr++;
					}

			// Allocating CPU profiler
			cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

			// FFT shift operation - CPU
			arr_3D_double = FFT::FFT_Shift_3D_double(arr_3D_double, size_X, size_Y, size_Z, cpuProfile);

			// Printing CPU output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_double[i][j][k]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 2), arr_3D_double[i][j][k]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 2), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 2), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 2), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(5, ((iLoop * 4 ) + 2), cpuProfile->unit_Sec);

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
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_double[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 3), arr_3D_flat_double[ctr]);
						ctr++;
					}

			// Uploading input array
			cuUtils::upload_3D_double(arr_3D_flat_double, in_dev_arr_3D_flat_double, size_X, size_Y, size_Z);

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

			// FFT shift
			cuFFTShift_3D_Double(cuBlock, cuGrid, out_dev_arr_3D_flat_double, in_dev_arr_3D_flat_double, size_X, cuProfile);

			// Clear the array to make sure that the results are true
			Array::zeroArray_3D_flat_double(arr_3D_flat_double , size_X, size_Y, size_Z);

			// Downloading output array
			cuUtils::download_3D_double(arr_3D_flat_double, out_dev_arr_3D_flat_double, size_X, size_Y, size_Z);

			// Free device memory
			cutilSafeCall(cudaFree((void*) in_dev_arr_3D_flat_double));
			cutilSafeCall(cudaFree((void*) out_dev_arr_3D_flat_double));

			// Printing output
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						if(ctr == 0)
							xlSheet->writeNum(4 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_double[ctr]);
						if(ctr == size_X * size_Y * size_Z - 1)
							xlSheet->writeNum(5 + START_ROW_DATA, ((iLoop * 4 ) + 4), arr_3D_flat_double[ctr]);
						ctr++;
					}

			// Printing profile data
			xlSheet->writeNum(2, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(3, ((iLoop * 4 ) + 4), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(4, ((iLoop * 4 ) + 4), cuProfile->kernelDuration );
			xlSheet->writeNum(5, ((iLoop * 4 ) + 4), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(6, ((iLoop * 4 ) + 4), cuProfile->kernelExecErr);

			// Adding the timing to the average profiler
			cuTotalProfile->kernelDuration += cuProfile->kernelDuration;

			// Dellocating: Host memory, profiles
			FREE_MEM_3D_DOUBLE(arr_3D_double, size_X, size_Y, size_Z);
			FREE_MEM_1D(arr_3D_flat_double);
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

