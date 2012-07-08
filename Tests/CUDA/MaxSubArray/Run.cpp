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

#define MAX_SIZE_1D 1024

#include "Array/Real/Array.h"
#include "ex_MaxSubArray.h"

int main(int argc, char** argv)
{
	// Number of numCores for OpenMP implementation
	int numCores = atoi (argv[1]);
	INFO("Number of available cores = " + ITS(numCores));

	// Number of iterations for gathering some statistics :)
	int numItr = 1;

	// Excel sheet parameters
	Book* xlBook;
	Sheet* xlSheet;

	/* xl Strings */
	char sizeString[25];
	char auxString[25];

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created");

	if(xlBook)
	{
		for (int arrSize_1D = 128; arrSize_1D <= MAX_SIZE_1D; arrSize_1D *= 2)
		{
			snprintf(sizeString, sizeof(sizeString), "%s", "");
			snprintf(auxString, sizeof(auxString), "%d", arrSize_1D);
			strcat(sizeString, auxString);

			// Create the corresponding XL sheet
			xlSheet = Utils::xl::addSheetToBook(sizeString, xlBook);

			if (xlSheet)
			{
				// Allocating the input array
				int* inputArray = (int*) malloc(sizeof(int) * arrSize_1D * arrSize_1D);
				INFO("Allocating input array with size = "
						+ ITS(arrSize_1D) + "x" + ITS(arrSize_1D));

				INFO("Array size = " + ITS(arrSize_1D) + "x" + ITS(arrSize_1D));
				// Filling the array with random numbers
				Array::fillArray_2D_flat_int(inputArray, arrSize_1D, arrSize_1D, 0);

				// Do the CPU implementation with OpenMP
				INFO("CPU implementation with OpenMP");
				ex_MaxSubArray::getMax_CPU(inputArray, numCores, arrSize_1D, arrSize_1D, numItr, xlSheet);

				SEP();

				// GPU implementation CUDA
				INFO("GPU implementation with CUDA");
				ex_MaxSubArray::getMax_CUDA(inputArray, arrSize_1D, arrSize_1D, numItr, xlSheet);

				// Freeing the memory
				FREE_MEM_1D(inputArray);
			}
			else
			{
				INFO("Invalid XL sheet was created. Exiting ...");
				EXIT(0);
			}
		}
	}
	else
	{
		INFO("Invalid XL book was created. Exiting ...");
		EXIT(0);
	}

	// Writhe the Xl book to disk
	snprintf(sizeString, sizeof(sizeString), "%s", "");
	strcat(sizeString, "MaxSubArray_");
	snprintf(auxString, sizeof(auxString), "%d", numCores);
	strcat(sizeString, auxString);
	strcat(sizeString, ".xls");

	Utils::xl::saveBook(xlBook, sizeString);
	INFO("Saving xlBook to disk");

	// Release the book to be able to OPEN it
	Utils::xl::releaseBook(xlBook);
	INFO("Releasing xlBook");

	return 0;
}

// Read the input image files
// for(int argFile = 2; argFile < argc; argFile++)
// INFO("Reading the input file");
// char* fileName = argv[argFile];
// ex_MaxSubArray::readFile(fileName, inputArray, numRows, numCols);
