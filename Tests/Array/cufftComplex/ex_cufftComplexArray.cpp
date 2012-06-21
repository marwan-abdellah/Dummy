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

#include "ex_cufftComplexArray.h"
#include "Utilities/XL.h"
#include "Array/Complex/cuComplex.h"
#include <cufft.h>
#include "Utilities/Memory.h"
#include "Utilities/MACROS.h"

namespace ex_cufftComplexArray
{
	int size_X = 8;
	int size_Y = 8;
	int size_Z = 8;
	int size_N;
	int ctr = 0;

	Book* xlBook;
	Sheet* xlSheet;

	char* xlBookName_Single = "ex_cufftComplexArray_Single.xls";
	char* xlBookName_Double= "ex_cufftComplexArray_Double.xls";

	cufftComplex* cuCmplxArray_1D;
	cufftComplex* cuCmplxArray_2D_flat;
	cufftComplex* cuCmplxArray_3D_flat;
	cufftComplex** cuCmplxArray_2D;
	cufftComplex*** cuCmplxArray_3D;

	cufftDoubleComplex* cuDblCmplxArray_1D;
	cufftDoubleComplex* cuDblCmplxArray_2D_flat;
	cufftDoubleComplex* cuDblCmplxArray_3D_flat;
	cufftDoubleComplex** cuDblCmplxArray_2D;
	cufftDoubleComplex*** cuDblCmplxArray_3D;
}

void ex_cufftComplexArray::DumpNumbers_Single()
{
	INFO("ex_cufftComplexArray - Single Precision");

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created : " + CATS(xlBookName_Single));
	SEP();

	if(xlBook)
	{
		/**********************************************************
		 * 1D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("1D", xlBook);

		if (xlSheet)
		{
			INFO("** 1D Case");

			size_N = size_X;

			// Allocation
			cuCmplxArray_1D = MEM_ALLOC_1D(cufftComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::cuComplex::fillArray_1D(cuCmplxArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuCmplxArray_1D[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuCmplxArray_1D[j].y);
			}

			// Filling array (Random)
			Array::cuComplex::fillArray_1D(cuCmplxArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuCmplxArray_1D[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuCmplxArray_1D[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuCmplxArray_1D);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}

		SEP();

		/**********************************************************
		 * 2D Flat Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("Flat_2D", xlBook);

		if (xlSheet)
		{
			INFO("** 2D Flat Case");

			size_N = size_X * size_Y;

			// Allocation
			cuCmplxArray_2D_flat = MEM_ALLOC_1D(cufftComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::cuComplex::fillArray_2D_flat(cuCmplxArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuCmplxArray_2D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuCmplxArray_2D_flat[j].y);
			}

			// Filling array (Random)
			Array::cuComplex::fillArray_2D_flat(cuCmplxArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuCmplxArray_2D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuCmplxArray_2D_flat[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuCmplxArray_2D_flat);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 3D Flat Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("Flat_3D", xlBook);

		if (xlSheet)
		{
			INFO("** 3D Flat Case");

			size_N = size_X * size_Y * size_Z;

			// Allocation
			cuCmplxArray_3D_flat = MEM_ALLOC_1D(cufftComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuComplex::fillArray_3D_flat(cuCmplxArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuCmplxArray_3D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuCmplxArray_3D_flat[j].y);
			}

			// Filling array (Random)
			Array::cuComplex::fillArray_3D_flat(cuCmplxArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuCmplxArray_3D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuCmplxArray_3D_flat[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuCmplxArray_3D_flat);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 2D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("2D", xlBook);

		if (xlSheet)
		{
			INFO("** 2D Case");

			// Allocation
			cuCmplxArray_2D = MEM_ALLOC_2D_CUFFTCOMPLEX(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuComplex::fillArray_2D(cuCmplxArray_2D, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// Real component
					xlSheet->writeNum((ctr) + 2, 0, cuCmplxArray_2D[i][j].x);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 1, cuCmplxArray_2D[i][j].y);
				}

			// Filling array (Random)
			Array::cuComplex::fillArray_2D(cuCmplxArray_2D, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// Real component
					xlSheet->writeNum((ctr) + 2, 2, cuCmplxArray_2D[i][j].x);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 3, cuCmplxArray_2D[i][j].y);
				}

			// Freeing memory
			FREE_MEM_2D_CUFFTCOMPLEX(cuCmplxArray_2D, size_X, size_Y);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 3D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("3D", xlBook);

		if (xlSheet)
		{
			INFO("** 3D Case");

			// Allocation
			cuCmplxArray_3D = MEM_ALLOC_3D_CUFFTCOMPLEX(size_X, size_Y, size_Z);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuComplex::fillArray_3D(cuCmplxArray_3D, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// Real component
						xlSheet->writeNum((ctr) + 2, 0, cuCmplxArray_3D[i][j][k].x);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 1, cuCmplxArray_3D[i][j][k].y);
					}

			// Filling array (Random)
			Array::cuComplex::fillArray_3D(cuCmplxArray_3D, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// Real component
						xlSheet->writeNum((ctr) + 2, 2, cuCmplxArray_3D[i][j][k].x);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 3, cuCmplxArray_3D[i][j][k].y);
					}

			// Freeing memory
			FREE_MEM_3D_CUFFTCOMPLEX(cuCmplxArray_3D, size_X, size_Y, size_Z);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		INFO("Sheets created & filled with data");

		// Writhe the Xl book to disk
		Utils::xl::saveBook(xlBook, xlBookName_Single);
		INFO("Saving " + CATS(xlBookName_Single) + " to disk")

		// Release the book to be able to OPEN it
		Utils::xl::releaseBook(xlBook);
		INFO("Releasing xlBook");
	}
	else
	{
		INFO("No valid xlBook was created, EXITTING ...");
		EXIT(0);
	}
}

void ex_cufftComplexArray::DumpNumbers_Double()
{
	INFO("ex_cufftComplexArray - Double Precision");

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created : " + CATS(xlBookName_Double));
	SEP();

	if(xlBook)
	{
		/**********************************************************
		 * 1D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("1D", xlBook);

		if (xlSheet)
		{
			INFO("** 1D Case");

			size_N = size_X;

			// Allocation
			cuDblCmplxArray_1D = MEM_ALLOC_1D(cufftDoubleComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::cuDoubleComplex::fillArray_1D(cuDblCmplxArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuDblCmplxArray_1D[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuDblCmplxArray_1D[j].y);
			}

			// Filling array (Random)
			Array::cuDoubleComplex::fillArray_1D(cuDblCmplxArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuDblCmplxArray_1D[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuDblCmplxArray_1D[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuDblCmplxArray_1D);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}

		SEP();

		/**********************************************************
		 * 2D Flat Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("Flat_2D", xlBook);

		if (xlSheet)
		{
			INFO("** 2D Flat Case");

			size_N = size_X * size_Y;

			// Allocation
			cuDblCmplxArray_2D_flat = MEM_ALLOC_1D(cufftDoubleComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::cuDoubleComplex::fillArray_2D_flat(cuDblCmplxArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuDblCmplxArray_2D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuDblCmplxArray_2D_flat[j].y);
			}

			// Filling array (Random)
			Array::cuDoubleComplex::fillArray_2D_flat(cuDblCmplxArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuDblCmplxArray_2D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuDblCmplxArray_2D_flat[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuDblCmplxArray_2D_flat);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 3D Flat Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("Flat_3D", xlBook);

		if (xlSheet)
		{
			INFO("** 3D Flat Case");

			size_N = size_X * size_Y * size_Z;

			// Allocation
			cuDblCmplxArray_3D_flat = MEM_ALLOC_1D(cufftDoubleComplex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuDoubleComplex::fillArray_3D_flat(cuDblCmplxArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, cuDblCmplxArray_3D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, cuDblCmplxArray_3D_flat[j].y);
			}

			// Filling array (Random)
			Array::cuDoubleComplex::fillArray_3D_flat(cuDblCmplxArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, cuDblCmplxArray_3D_flat[j].x);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, cuDblCmplxArray_3D_flat[j].y);
			}

			// Freeing memory
			FREE_MEM_1D(cuDblCmplxArray_3D_flat);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 2D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("2D", xlBook);

		if (xlSheet)
		{
			INFO("** 2D Case");

			// Allocation
			cuDblCmplxArray_2D = MEM_ALLOC_2D_CUFFTDOUBLECOMPLEX(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuDoubleComplex::fillArray_2D(cuDblCmplxArray_2D, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// Real component
					xlSheet->writeNum((ctr) + 2, 0, cuDblCmplxArray_2D[i][j].x);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 1, cuDblCmplxArray_2D[i][j].y);
				}

			// Filling array (Random)
			Array::cuDoubleComplex::fillArray_2D(cuDblCmplxArray_2D, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
				{
					// Real component
					xlSheet->writeNum((ctr) + 2, 2, cuDblCmplxArray_2D[i][j].x);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 3, cuDblCmplxArray_2D[i][j].y);
				}

			// Freeing memory
			FREE_MEM_2D_CUFFTDOUBLECOMPLEX(cuDblCmplxArray_2D, size_X, size_Y);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		/**********************************************************
		 * 3D Case
		 **********************************************************/
		xlSheet = Utils::xl::addSheetToBook("3D", xlBook);

		if (xlSheet)
		{
			INFO("** 3D Case");

			// Allocation
			cuDblCmplxArray_3D = MEM_ALLOC_3D_CUFFTDOUBLECOMPLEX(size_X, size_Y, size_Z);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::cuDoubleComplex::fillArray_3D(cuDblCmplxArray_3D, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// Real component
						xlSheet->writeNum((ctr) + 2, 0, cuDblCmplxArray_3D[i][j][k].x);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 1, cuDblCmplxArray_3D[i][j][k].y);
					}

			// Filling array (Random)
			Array::cuDoubleComplex::fillArray_3D(cuDblCmplxArray_3D, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
					{
						// Real component
						xlSheet->writeNum((ctr) + 2, 2, cuDblCmplxArray_3D[i][j][k].x);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 3, cuDblCmplxArray_3D[i][j][k].y);
					}

			// Freeing memory
			FREE_MEM_3D_CUFFTDOUBLECOMPLEX(cuDblCmplxArray_3D, size_X, size_Y, size_Z);
			INFO("Freeing memory");
		}
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}
		SEP();

		INFO("Sheets created & filled with data");

		// Writhe the Xl book to disk
		Utils::xl::saveBook(xlBook, xlBookName_Double);
		INFO("Saving " + CATS(xlBookName_Double) + " to disk")

		// Release the book to be able to OPEN it
		Utils::xl::releaseBook(xlBook);
		INFO("Releasing xlBook");
	}
	else
	{
		INFO("No valid xlBook was created, EXITTING ...");
		EXIT(0);
	}
}
