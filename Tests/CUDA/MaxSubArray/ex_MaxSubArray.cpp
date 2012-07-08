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

#include "ex_MaxSubArray.h"
#include "FFT/FFTShift.h"
#include "CUDA/Utilities/cuUtils.h"

#include "Utilities/Utils.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/cuExterns.h"
#include "cuExternsTest.h"

using std::cout;
using std::endl;
using std::string;

namespace ex_MaxSubArray
{
	/* @ Profilers */
	cudaProfile* cuProfile;
	durationStruct* cpuProfile;

	cudaProfile* cuTotalProfile;
	durationStruct* cpuTotalProfile;
}

void ex_MaxSubArray::readFile(char* fileName, int* inputArray, int numRows, int numCols)
{
	INFO("Reading file - Starting");

	// Array indicies
	int xIdx = 0;
	int yIdx = 0;

	// Input stream
	std::ifstream inStream(fileName);

	if (inStream)
	{
		// Reading lineRow by lineRow
		std::string lineRow;

		// Initializing the Y index
		yIdx = 0;

		// Getting line by line
		while (std::getline(inStream, lineRow))
		{
			// Getting column by column
			std::stringstream split(lineRow);

			int inputVal;

			// Resetting the X index
			xIdx = 0;
			while (split >> inputVal)
			{
				// storing the input value from the file to the array
				inputArray[((yIdx * numRows) + xIdx)] = inputVal;

				// Incrementing the X idex
				xIdx++;
			}

			// Incrementing the y index
			yIdx++;
		}
	}

	// Closing the input stream
	INFO("Closing the input stream");
	inStream.close();

	INFO("Reading file - Done");
}

/*
 * This functions is divided in 2 stages. For the time being, we
 * will refer to them by STAGE_1 & STAGE_2.
 */
void ex_MaxSubArray::getMax_CPU(int* inputArray, int numCores, int numRows, int numCols, int numItr, Sheet* xlSheet)
{
	INFO("Starting CPU implementation : Iterations " + ITS(numItr));

	/* CPU timing parameters */
	time_boost start, end;

	// Allocating CPU profiler
	cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
	cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

	if (xlSheet)
	{
		// Averaging Rows
		xlSheet->writeStr(13, (6), "Avg");
		xlSheet->writeStr(15, (5), "ns");
		xlSheet->writeStr(16, (5), "us");
		xlSheet->writeStr(17, (5), "ms");
		xlSheet->writeStr(18, (5), "s");

		// Averaging Headers
		xlSheet->writeStr(14, (6), "S1_CPU");
		xlSheet->writeStr(14, (7), "S1_GPU");
		xlSheet->writeStr(14, (8), "S2_CPU");
		xlSheet->writeStr(14, (9), "S2_GPU");
		xlSheet->writeStr(14, (10), "T_CPU");
		xlSheet->writeStr(14, (11), "T_GPU");

		// Rows
		xlSheet->writeStr(3, (0), "# Itr");
		xlSheet->writeStr(3, (0), "ns");
		xlSheet->writeStr(4, (0), "us");
		xlSheet->writeStr(5, (0), "ms");
		xlSheet->writeStr(6, (0), "s");
		xlSheet->writeStr(7, (0), "cuErrors");

		// Iterate to average the results
		for (int itr = 0; itr < numItr; itr++)
		{
			// Headers
			xlSheet->writeNum(1, ((itr * 6) + 1), itr);
			xlSheet->writeStr(2, ((itr * 6) + 1), "S1_CPU");
			xlSheet->writeStr(2, ((itr * 6) + 3), "S2_CPU");
			xlSheet->writeStr(2, ((itr * 6) + 5), "T_CPU");

			/*
			 * An array for holding the maximum values of all
			 * possible combination
			 */
			Max maxValues[numRows];

			/*
			 * Start of parallel region inStream which we are going
			 * to divide numRows on the number of threads, each thread
			 * will calculate the maximum of all possible combination
			 * and only store the maximum of them all inStream maxVal
			 */
#pragma omp parallel num_threads(numCores)
			{
				// Intermediate parameters
				int tempMaxSum = 0;
				int candMaxSubArr = 0 ,j;

				// Array prefSum will be used to calculate the prefix sum
				int prefSum[numCols];

				// @ STAGE_1 "Starting"
				start = Timers::BoostTimers::getTime_MicroSecond();

#pragma omp for schedule(dynamic)
				for(int g = 0; g < numRows; g++)
				{
					// Resetting the max value to 0
					maxValues[g].val = 0;

					// Resetting the prefix sum array
					for(int iCtr = 0; iCtr < numCols; iCtr++)
						prefSum[iCtr] = 0;

					// Iterating
					// TODO: To document what is happening in each iteration
					for(int i = g; i < numRows; i++)
					{
						tempMaxSum = 0;
						j = 0;
						for(int h = 0; h < numCols; h++)
						{
							prefSum[h] = prefSum[h] + inputArray[i*numRows+h];
							// t is the prefix sum of the strip start at row z to row xIdx
							tempMaxSum = tempMaxSum + prefSum[h];

							if( tempMaxSum > candMaxSubArr)
							{
								candMaxSubArr = tempMaxSum;
								maxValues[g].val = candMaxSubArr;
								maxValues[g].x1 = g;
								maxValues[g].y1 = j;
								maxValues[g].x2 = i;
								maxValues[g].y2 = h;
							}

							if( tempMaxSum < 0 )
							{
								tempMaxSum = 0;
								j = h + 1;
							}
						}
					}
				}
			}

			// @ STAGE_1 "Done"
			end = Timers::BoostTimers::getTime_MicroSecond();

			// Calculate the duration of STAGE_1
			cpuProfile = Timers::BoostTimers::getDuration(start, end);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 1), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(4, ((itr * 6) + 1), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(5, ((itr * 6) + 1), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(6, ((itr * 6) + 1), cpuProfile->unit_Sec);

			int selectedMAxVal = 0;
			int indexMaxValue=0;

			// @ STAGE_2 "Starting"
			start = Timers::BoostTimers::getTime_MicroSecond();

			// Search for the maximum inputVal inStream all maximum candidates
			for (int i = 0; i < numRows; i++)
			{
				if (maxValues[i].val >selectedMAxVal)
				{
					selectedMAxVal = maxValues[i].val;
					indexMaxValue = i;
				}
			}

			// @ STAGE_2 "Done"
			end = Timers::BoostTimers::getTime_MicroSecond();

			// Calculate the duration for STAGE_2
			cpuProfile = Timers::BoostTimers::getDuration(start, end);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 3), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(4, ((itr * 6) + 3), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(5, ((itr * 6) + 3), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(6, ((itr * 6) + 3), cpuProfile->unit_Sec);

			xlSheet->writeNum(8, ((itr * 6) + 3), maxValues[indexMaxValue].y1);
			xlSheet->writeNum(9, ((itr * 6) + 3), maxValues[indexMaxValue].x1);
			xlSheet->writeNum(10, ((itr * 6) + 3), maxValues[indexMaxValue].y2);
			xlSheet->writeNum(11, ((itr * 6) + 3), maxValues[indexMaxValue].x2);
		}
	}
	else
	{
		INFO("No valid XL sheet was created. Exiting ... ");
		EXIT(0);
	}

	FREE_MEM_1D(cpuProfile);
	INFO("CPU implementation - Done");
}

/*
 * This functions is divided in 2 stages. For the time being, we
 * will refer to them by STAGE_1 & STAGE_2.
 */
void ex_MaxSubArray::getMax_CUDA(int* hostInputArray, int numRows, int numCols, int numItr, Sheet* xlSheet)
{
	INFO("Starting CUDA implementation");

	// Allocating the CUDA profiler
	cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

	if (xlSheet)
	{
		for (int itr = 0; itr < numItr; itr++)
		{
			// Headers
			xlSheet->writeStr(2, ((itr * 6) + 2), "S1_GPU");
			xlSheet->writeStr(2, ((itr * 6) + 4), "S2_GPU");
			xlSheet->writeStr(2, ((itr * 6) + 6), "T_GPU");

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);


			// Memory required for input & output arrays
			INFO("Calculating memory required");
			const int inputArraySize = sizeof(int) * numRows * numCols;
			const int outputArrySize = sizeof(Max) * numRows;

			// Input & output arrays on the device side
			int* devInputArray;
			Max *devMaxValues;

			// Allocate an array to hold the maximum of all possible combination
			Max hostMaxValues[numRows];

			// Allocating the device arrays
			INFO("Allocating device arrays");
			cutilSafeCall(cudaMalloc((void**)&devInputArray, inputArraySize));
			cutilSafeCall(cudaMalloc((void**)&devMaxValues, outputArrySize));

			// Upload the input array to the device side
			INFO("Uploading the input array to the GPU");
			cutilSafeCall(cudaMemcpy(devInputArray, hostInputArray, inputArraySize, cudaMemcpyHostToDevice));

			// Configuring the GPU
			INFO("Addjusting Gridding configuration");
			dim3 cuBlock(128, 1, 1);
			dim3 cuGrid(numRows/cuBlock.x, 1, 1);

			// Invokig the CUDA kernel
			INFO("Invoking CUDA kernel");
			cuGetMax(cuBlock, cuGrid, devMaxValues, devInputArray, numRows, numCols, cuProfile);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 2), cuProfile->kernelDuration * 1000 * 1000);
			xlSheet->writeNum(4, ((itr * 6) + 2), cuProfile->kernelDuration * 1000);
			xlSheet->writeNum(5, ((itr * 6) + 2), cuProfile->kernelDuration );
			xlSheet->writeNum(6, ((itr * 6) + 2), cuProfile->kernelDuration / 1000);
			xlSheet->writeNum(7, ((itr * 6) + 2), cuProfile->kernelExecErr);

			// Checking if kernel execution failed or not
			cutilCheckMsg("Kernel execution failed \n");

			// Download the maxValues array to the host side
			INFO("Downloading the resulting array to the CPU");
			cutilSafeCall(cudaMemcpy(hostMaxValues, devMaxValues, outputArrySize, cudaMemcpyDeviceToHost));

			// Freeingthe allocated memory on the device
			INFO("Freeing the device memory");
			cudaFree(devMaxValues);

			int selectedMaxVal = 0;
			int indexMaxVal = 0;

			// Search for the maximum value in all maximum candidates
			for (int i = 0; i < numRows; i++)
			{
				if (hostMaxValues[i].val > selectedMaxVal)
				{
					// Updating the selected values
					selectedMaxVal = hostMaxValues[i].val;

					// Updating the index
					indexMaxVal = i;
				}
			}

			xlSheet->writeNum(8, ((itr * 6) + 3), hostMaxValues[indexMaxVal].y1);
			xlSheet->writeNum(9, ((itr * 6) + 4), hostMaxValues[indexMaxVal].x1);
			xlSheet->writeNum(10, ((itr * 6) + 4), hostMaxValues[indexMaxVal].y2);
			xlSheet->writeNum(11, ((itr * 6) + 4), hostMaxValues[indexMaxVal].x2);
		}
	}
	FREE_MEM_1D(cuProfile);
	INFO("CUDA implementation Done");
}
