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

#define MAX_SIZE_1D 2048

#include "Array/Real/Array.h"
#include "ex_MaxSubArray.h"

int main(int argc, char** argv)
{
	// Number of numCores for OpenMP implementation
	int numCores = atoi (argv[1]);
	INFO("Number of available cores = " + ITS(numCores));

	// Excel sheet parameters
	Book* xlBook;
	Sheet* xlSheet;

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created");

	if(xlBook)
	{
		for (int itrSize = 128; itrSize < MAX_SIZE_1D; itrSize *= 2)
		{
			// Allocating the input array
			int* inputArray = (int*) malloc(sizeof(int) * itrSize * itrSize);
			INFO("Allocating input array with size = "
					+ ITS(itrSize) + "x" + ITS(itrSize));

			INFO("Array size = " + ITS(itrSize) + "x" + ITS(itrSize));
			// Filling the array with random numbers
			Array::fillArray_2D_flat_int(inputArray, itrSize, itrSize, 0);

			// Do the CPU implementation with OpenMP
			INFO("CPU implementation with OpenMP");
			ex_MaxSubArray::getMax_CPU(inputArray, numCores, itrSize, itrSize);

			SEP();

			// Allocate an array to hold the maximum of all possible combination
			Max host_maxValues[itrSize];

			// GPU implementation CUDA
			INFO("GPU implementation with CUDA");
			ex_MaxSubArray::getMax_CUDA(inputArray, host_maxValues, itrSize, itrSize);

			// Freeing the memory
			FREE_MEM_1D(inputArray);
		}
	}

	return 0;
}

// Read the input image files
// for(int argFile = 2; argFile < argc; argFile++)
// INFO("Reading the input file");
// char* fileName = argv[argFile];
// ex_MaxSubArray::readFile(fileName, inputArray, numRows, numCols);
