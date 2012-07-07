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

void ex_MaxSubArray::getMax_CPU(int* inputArray,int numCores, int numRows, int numCols)
{
	INFO("Starting CPU implementation");
	/*
	 * An array for holding the maximum values of all
	 * possible combination
	 */
	Max maxValues[numRows];

	/*
	 * start of parallel region inStream which we are going
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
					tempMaxSum = tempMaxSum + prefSum[h]; // t is the prefix sum of the strip start at row z to row xIdx

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

	int selectedMAxVal = 0;
	int indexMaxValue=0;

	// Search for the maximum inputVal inStream all maximum candidates
	for (int i = 0; i < numRows; i++)
	{
		if (maxValues[i].val >selectedMAxVal)
		{
			selectedMAxVal = maxValues[i].val;
			indexMaxValue = i;
		}
	}

	INFO("CPU results for the Max Sub-Array : " + CATS("[") +
		ITS(maxValues[indexMaxValue].y1) + "," +
		ITS(maxValues[indexMaxValue].x1) + "," +
		ITS(maxValues[indexMaxValue].y2) + "," +
		ITS(maxValues[indexMaxValue].x2) + CATS("]"))

	INFO("CPU implementation - Done");
}

void ex_MaxSubArray::getMax_CUDA(int* hostInputArray, Max* hostMaxValues, int numRows, int numCols)
{
	INFO("Starting CUDA implementation");

	// Memory required for input & output arrays
	INFO("Calculating memory required");
	const int inputArraySize = sizeof(int) * numRows * numCols;
	const int outputArrySize = sizeof(Max) * numRows;

	// Input & output arrays on the device side
	int* devInputArray;
	Max *devMaxValues;

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

	// Allocating the CUDA profiler
	cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

	// Invokig the CUDA kernel
	INFO("Invoking CUDA kernel");
	cuGetMax(cuBlock, cuGrid, devMaxValues, devInputArray, numRows, numCols, cuProfile);

    // Checking if kernel execution failed or not
    cutilCheckMsg("Kernel execution failed \n");

    // Download the maxValues array to the host side
    INFO("Downloading the resulting array to the CPU");
    cutilSafeCall(cudaMemcpy(hostMaxValues, devMaxValues, outputArrySize, cudaMemcpyDeviceToHost));

	INFO("GPU Benchmarks: "
			"\n \t Nano-Sec : " + DTS(cuProfile->kernelDuration * 1000 * 1000) +
			"\n \t Micro-Sec : " + DTS(cuProfile->kernelDuration * 1000) +
			"\n \t Milli-Sec : " + DTS(cuProfile->kernelDuration) +
			"\n \t Sec : " + DTS(cuProfile->kernelDuration / 1000));

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

			// updating the index
			indexMaxVal = i;
		}
	}

	INFO("GPU results for the Max Sub-Array : " + CATS("[") +
			ITS(hostMaxValues[indexMaxVal].y1) + "," +
			ITS(hostMaxValues[indexMaxVal].x1) + "," +
			ITS(hostMaxValues[indexMaxVal].y2) + "," +
			ITS(hostMaxValues[indexMaxVal].x2) + CATS("]"))

	INFO("CUDA implementation Done");
}
