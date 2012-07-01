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
 * Note(s)      : Maximum Block Size = [512][512][64] in 1D
 * * cudaErrorInvalidConfiguration
			 * This indicates that a kernel launch is requesting resources
			 * that can never be satisfied by the current device.
			 * Requesting more shared memory per block than the
			 * device supports will trigger this error, as will
			 * requesting too many threads or blocks. See cudaDeviceProp
			 * for more device limitations.
 *********************************************************************/

/* @ Maximum dimensions in 3D */
#define N_3D_MAX 128

/* @ Maximum CUDA block dimensions in 3D */
#define MAX_BLOCK_SIZE_X 16
#define MAX_BLOCK_SIZE_Y 16
#define MAX_BLOCK_SIZE_Z 1

#define MAX_GRID_SIZE_X 65535
#define MAX_GRID_SIZE_Y 65535
#define MAX_GRID_SIZE_Z 1

/* @ Considering the UNIFIED 3D case*/
#define MAX_BLOXK_SIZE_N 16

#include "iB_Complex_FFTShift_3D.h"

int main()
{
	/* @ Iterations "for averaging" */
	int nLoop = ITERATIONS;

	Book* xlBook;
	Sheet* xlSheet;

	/* @ CUDA configuration parameters */
	dim3 cuBlock;
	dim3 cuGrid;

	/* @ To terminate the loop if INVALID configuration */
	int cuFlag = 1;
	int blkFLag = 1;

	/* @ Unified dimensions in X, Y, Z */
	int N = 2;
	char sizeString[25];
	char auxString[25];

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created");

	if(xlBook)
	{
		/**********************************************************
		 * Float Case
		 **********************************************************/
		for (int i = N; i <= N_3D_MAX; i *= 2)
		{
			/*
			 * Calculate CUDA iterations according to array size
			 * and maximum block dimensions.
			 */

			/* @ Initial cuBlock with minimum configuration */
			cuBlock.x = 2;
			cuBlock.y = 2;

			// Reset cuFlag
			cuFlag = 1;

			// Clearing the sizeString
			snprintf(sizeString, sizeof(sizeString), "%s", "");

			while ( cuFlag )
			{
				// Clearing the sizeString
				snprintf(sizeString, sizeof(sizeString), "%s", "");

				cuGrid.x = i / cuBlock.x;
				cuGrid.y = i / cuBlock.y;


				if (!(cuBlock.x <= MAX_BLOCK_SIZE_X &&
					cuBlock.y <= MAX_BLOCK_SIZE_Y 	&&
					cuGrid.x > 0 					&&
					cuGrid.y > 0					&&
					cuGrid.x <= MAX_GRID_SIZE_X	&&
					cuGrid.y <= MAX_GRID_SIZE_Y
					))
					cuFlag = 0;

				// Copy the size string with < Arr[]_Grid[] >
				strcat(sizeString, "Arr");
				snprintf(auxString, sizeof(auxString), "%d", i);
				strcat(sizeString, auxString);
				strcat(sizeString, "_Blk");
				snprintf(auxString, sizeof(auxString), "%d", cuBlock.x);
				strcat(sizeString, auxString);

				if (cuFlag)
				{
					// Create the corresponding XL sheet
					xlSheet = Utils::xl::addSheetToBook(sizeString, xlBook);

					if (xlSheet)
						iB_Complex_FFTShift_3D::FFTShift_3D_Float_Seq(i, i, i, xlSheet, nLoop, cuGrid, cuBlock);
				}

				/* Next block dimensions */
				cuBlock.x *= 2;
				cuBlock.y *= 2;
			}
		}
	}

	// Writhe the Xl book to disk
	Utils::xl::saveBook(xlBook, "iB_ComplexSingleSeq_FFTShift_3D.xls");
	INFO("Saving xlBook to disk")

	// Release the book to be able to OPEN it
	Utils::xl::releaseBook(xlBook);
	INFO("Releasing xlBook");

	return 0;
}
