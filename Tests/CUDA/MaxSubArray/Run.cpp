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
	for(int argFile = 2; argFile < argc; argFile++)
	{
		char* fileName = argv[argFile];
		ex_MaxSubArray::readFile(fileName, inputArray);
		INFO("Reading the input file");

		// Do the CPU implementation with OpenMP
		INFO("CPU implementation with OpenMP");
		ex_MaxSubArray::getMax_CPU(inputArray, numCores);

		SEP();

		// Allocate an array to hold the maximum of all possible combination
		Max host_maxValues[ex_MaxSubArray::numRows];

		// GPU implementation CUDA
		INFO("GPU implementation with CUDA");
		ex_MaxSubArray::getMax_CUDA(inputArray, host_maxValues);

		int selectedMaxVal = 0;
		int indexMaxVal = 0;

		// Search for the maximum value in all maximum candidates
		for (int i = 0; i < ex_MaxSubArray::numRows; i++)
		{
			if (host_maxValues[i].val > selectedMaxVal)
			{
				// Updating the selected values
				selectedMaxVal = host_maxValues[i].val;

				// updating the index
				indexMaxVal = i;
			}
		}

		INFO("GPU results for the Max Sub-Array : " + CATS("[") +
				ITS(host_maxValues[indexMaxVal].y1) + "," +
				ITS(host_maxValues[indexMaxVal].x1) + "," +
				ITS(host_maxValues[indexMaxVal].y2) + "," +
				ITS(host_maxValues[indexMaxVal].x2) + CATS("]"))
	}

	FREE_MEM_1D(inputArray);
	return 0;
}
