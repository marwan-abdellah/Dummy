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
 * Note(s)      : Maximum Block Size = [512][512][64]
 *********************************************************************/

/* @ Maximum dimensions in 2D */
#define N_2D_MAX 1024

/* @ Maximum CUDA block dimensions */
#define MAX_BLOCK_SIZE_X 512
#define MAX_BLOCK_SIZE_Y 512
#define MAX_BLOCK_SIZE_Z 64

/* @ Considering the UNIFIED 2D case*/
#define MAX_BLOXK_SIZE_N 512

#include "iB_cuFFTShift_2D.h"

int main()
{
	/* @ Iterations "for averaging" */
	int nLoop = 1;

	Book* xlBook;
	Sheet* xlSheet;

	/* @ CUDA configuration parameters */
	dim3 cuBlock;
	dim3 cuGrid;

	/* @ To terminate the loop if INVALID configuration */
	bool cuFlag = 1;

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
		for (int i = N; i <= N_2D_MAX; i *= 2)
		{
			/*
			 * Calculate CUDA iterations according to array size
			 * and maximum block dimensions.
			 */

			/* @ Initial cuBlock with minimum configuration */
			cuBlock.x = 2;
			cuBlock.y = 2;
			cuBlock.z = 1;

			/* @ Initial cuGrid with SINGLE configuration */
			cuGrid.x = 1;
			cuGrid.y = 1;
			cuGrid.z = 1;

			cuFlag = 1;

			// Clearing the sizeString
			snprintf(sizeString, sizeof(sizeString), "%s", "");

			while ( cuFlag )
			{
				// Clearing the sizeString
				snprintf(sizeString, sizeof(sizeString), "%s", "");

				cuGrid.x = i / cuBlock.x;
				cuGrid.y = i / cuBlock.y;

				if (cuBlock.x <= MAX_BLOCK_SIZE_X 	&&
					cuBlock.y <= MAX_BLOCK_SIZE_Y 	&&
					cuBlock.z <= MAX_BLOCK_SIZE_Z	&&
					cuGrid.x > 0 					&&
					cuGrid.y > 0 					&&
					cuGrid.z > 0)
					cuFlag = 1;
				else
					cuFlag = 0;

				// Copy the size string with < Arr[]_Grid[] >
				strcat(sizeString, "Arr");
				snprintf(auxString, sizeof(auxString), "%d", i);
				strcat(sizeString, auxString);
				strcat(sizeString, "_Grid");
				snprintf(auxString, sizeof(auxString), "%d", cuGrid.x);
				strcat(sizeString, auxString);

				if(cuFlag)
				{
					// Create the corresponding XL sheet
					xlSheet = Utils::xl::addSheetToBook(sizeString, xlBook);

					if (xlSheet)
					{
						// iB_cuFFTShift_2D::FFTShift_2D_Float(i, i, xlSheet, nLoop, cuGrid, cuBlock);
						iB_cuFFTShift_2D::FFTShift_2D_Float_CUDA(i, i, xlSheet, nLoop, cuGrid, cuBlock);
					}

					/* Next block dimensions */
					cuBlock.x *= 2;
					cuBlock.y *= 2;
				}
			}
		}
	}

	// Writhe the Xl book to disk
	Utils::xl::saveBook(xlBook, "iB_cuFFTShift_2D.xls");
	INFO("Saving xlBook to disk")

	// Release the book to be able to OPEN it
	Utils::xl::releaseBook(xlBook);
	INFO("Releasing xlBook");

	return 0;
}