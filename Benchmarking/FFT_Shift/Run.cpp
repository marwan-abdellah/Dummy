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

/*@ Maximum dimensions in 2D */
#define N_2D_MAX (4 * 1024)

#include "iB_FFTShift.h"

int main()
{
	Book* xlBook;
	Sheet* xlSheet;
	int nLoop = 1;

	/* @ Unified dimensions in X, Y, Z */
	int N = 512;
	char sizeString[25];

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
			// Copy the size string
			snprintf(sizeString, sizeof(sizeString), "%d", i);

			// Create the corresponding XL sheet
			xlSheet = Utils::xl::addSheetToBook(sizeString, xlBook);

			if (xlSheet)
			{
				iB_FFTShift::FFTShift_2D_Float(i, i, xlSheet, nLoop);
			}
		}
	}

	// Writhe the Xl book to disk
	Utils::xl::saveBook(xlBook, "iB_FFTShift_2D.xls");
	INFO("Saving xlBook to disk")

	// Release the book to be able to OPEN it
	Utils::xl::releaseBook(xlBook);
	INFO("Releasing xlBook");





	//iB_FFTShift::FFTShift_2D_Float(8, 8);
	//iB_FFTShift::FFTShift_2D_CUDA(8, 8);

	//iB_FFTShift::FFTShift_3D_CPU(4, 4, 4);
	//iB_FFTShift::FFTShift_3D_CUDA(4,4, 4);

	return 0;
}
