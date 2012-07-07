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

#include "CUDA/cuGlobals.h"
#include "CUDA/cuExterns.h"
#include "cuExternsTest.h"

using std::cout;
using std::endl;
using std::string;

namespace ex_MaxSubArray
{
	/* Profilers */
	durationStruct* duration;
}

void ex_MaxSubArray::readFile(char* fileName, int* inputArray)
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

void ex_MaxSubArray::getMax_CPU(int* inputArray,int numCores)
{
	INFO("CPU implementation - Starting");
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
		int pr [numCols];

#pragma omp for schedule(dynamic)
		for( int g = 0; g < numRows; g++)
		{
			maxValues[g].S = 0;

			//array pr will be used to calculate the prefix sum
			for(int h = 0; h < numCols; h++)
				pr[h] = 0;

			for(int i = g; i < numRows; i++)
			{
				tempMaxSum = 0;
				j = 0;
				for(int h = 0; h < numCols; h++)
				{
					pr[h] = pr[h] + inputArray[i*numRows+h];
					tempMaxSum = tempMaxSum + pr[h]; // t is the prefix sum of the strip start at row z to row xIdx

					if( tempMaxSum > candMaxSubArr)
					{ 
						candMaxSubArr = tempMaxSum;
						maxValues[g].S = candMaxSubArr;
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


	int S = 0,ind=0;
	// search for the maximum inputVal inStream all maximum candidates
	for (int i = 0; i < numRows; i++)
	{
		if (maxValues[i].S >S)
		{
			S = maxValues[i].S;
			ind=i;
		}
	}

	cout << maxValues[ind].y1 << " " << maxValues[ind].x1 << " " << maxValues[ind].y2 << " "  << maxValues[ind].x2 <<" "<< endl;
	INFO("CPU implementation - Done");
}

void ex_MaxSubArray::getMax_CUDA(int* hostInputArray, Max* hostMaxValues)
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
	dim3 cuBlock(256, 1, 1);
	dim3 cuGrid(numRows/cuBlock.x, 1, 1);

	// Invokig the CUDA kernel
	INFO("Invoking CUDA kernel");
	cuGetMax(cuBlock, cuGrid, devMaxValues, devInputArray, numRows, numCols);

    // Checking if kernel execution failed or not
    cutilCheckMsg("Kernel execution failed \n");

    // Download the maxValues array to the host side
    INFO("Downloading the resulting array to the CPU");
    cutilSafeCall(cudaMemcpy(hostMaxValues, devMaxValues, outputArrySize, cudaMemcpyDeviceToHost));

    // Freeingthe allocated memory on the device
    INFO("Freeing the device memory");
	cudaFree(devMaxValues);

	INFO("CUDA implementation Done");
}
