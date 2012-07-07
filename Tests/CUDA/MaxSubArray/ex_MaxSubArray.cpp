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

using std::cout;
using std::endl;
using std::string;

namespace ex_MaxSubArray
{
	/* Profilers */
	durationStruct* duration;
}

void ex_MaxSubArray::readFile(char* fname, int* arr)
{
	// init input file stream to read from the file
	std::ifstream in(fname);


	if (in) {
		std::string line;
		
		int y = 0;
		while (std::getline(in, line)) {
			
			// Break down the row into column values
			std::stringstream split(line);
			int value;

			int x = 0;
			while (split >> value)
			{
				arr[y*rows+x] = value;
				x++;
			}
			y++;
		}
		
	}
	in.close();

}

void ex_MaxSubArray::getMax_CPU(int* arr,int cores)
{

	// allocate an array to hold the maximum of all possible combination
	Max maxValues[rows];

	//start of parallel region in which we are going to divide rows on the number of threads ,
	//each thread will calculate the maximum of all possible combination and only store the maximum of them all in maxVal
	#pragma omp parallel num_threads(cores)
	{
		// this will be used in calculating the maximum
		int tempMaxSum = 0;
		int candMaxSubArr = 0 ,j;
		int pr [cols];
		#pragma omp for schedule(dynamic)
		for( int g = 0; g < rows; g++)
		{
			//array pr will be used to calculate the prefix sum
			for(int h = 0; h < cols; h++) 
				pr[h] = 0;

			for(int i = g; i < rows; i++)
			{
				tempMaxSum = 0;
				j = 0;
				for(int h = 0; h < cols; h++)
				{
					pr[h] = pr[h] + arr[i*rows+h]; 
					tempMaxSum = tempMaxSum + pr[h]; // t is the prefix sum of the strip start at row z to row x
			
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
	// search for the maximum value in all maximum candidates
	for (int i = 0; i < rows; i++)
	{
		if (maxValues[i].S >S)
		{
			S = maxValues[i].S;
			ind=i;
		}
	}

	cout << maxValues[ind].y1 << " " << maxValues[ind].x1 << " " << maxValues[ind].y2 << " "  << maxValues[ind].x2 <<" "<< endl;
}

void ex_MaxSubArray::getMax_CUDA(int* host_inputArray, Max* host_maxValues)
{
	const int mem1 = sizeof(int) * rows * cols;
	const int mem2 = sizeof(Max) * rows;
/*
	int* dev_inputArray;
	cutilSafeCall(cudaMalloc( (void**)&dev_inputArray, mem1 ));
	cutilSafeCall(cudaMemcpy(dev_inputArray, host_inputArray, mem1, cudaMemcpyHostToDevice));

	Max *dev_maxValues;
    cutilSafeCall(cudaMalloc( (void**)&dev_maxValues, mem2 ));

    findMax<<<1,rows>>>( rows, cols, dev_maxValues, dev_inputArray );
	cutilCheckMsg("Kernel execution failed\n");

    cutilSafeCall(cudaMemcpy( host_maxValues, dev_maxValues, mem2, cudaMemcpyDeviceToHost ));

	cudaFree( dev_maxValues );
*/
}
