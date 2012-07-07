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

int main(int argc, char** argv)
{
	// Allocating the input array
	int* inputArray = (int*) malloc(sizeof(int) * ex_MaxSubArray::numRows * ex_MaxSubArray::numCols);
	INFO("Allocating input array with size = "
			+ ITS(ex_MaxSubArray::numRows) + "x" + ITS(ex_MaxSubArray::numCols));

	// Number of numCores for OpenMP implementation
	int numCores = atoi (argv[1]);
	INFO("Number of available cores = " + ITS(numCores));

	// Read the input image files
	for(int argFile=2; argFile < argc; argFile++)
	{
		char* fileName = argv[argFile];
		ex_MaxSubArray::readFile(fileName, inputArray);
		INFO("Readig the input file");

		// Do the CPU implementation with OpenMP
		INFO("CPU implementation with OpenMP");
		ex_MaxSubArray::getMax_CPU(inputArray, numCores);
		
		// allocate an array to hold the maximum of all possible combination
		Max host_maxValues[ex_MaxSubArray::numRows];

		// GPU implementation CUDA
		INFO("GPU implementation with CUDA");
		ex_MaxSubArray::getMax_CUDA(inputArray, host_maxValues);

		int S = 0,ind=0;
		// search for the maximum value in all maximum candidates
		for (int i = 0; i < ex_MaxSubArray::numRows; i++)
		{
			if (host_maxValues[i].S >S)
			{
				S = host_maxValues[i].S;
				ind=i;
			}
		}

		cout << host_maxValues[ind].y1 << " " << host_maxValues[ind].x1 << " " << host_maxValues[ind].y2 << " "
			<< host_maxValues[ind].x2 <<" "<< endl;

	}

	free(inputArray);
	return 0;
}
