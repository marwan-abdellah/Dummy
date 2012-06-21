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

#include "ex_fftwComplexArray.h"
#include "Utilities/XL.h"
#include "Array/Complex/fftwComplex.h"
#include <cufft.h>
#include "Utilities/Memory.h"
#include "Utilities/MACROS.h"

namespace ex_fftwComplexArray
{
	int size_X = 8;
	int size_Y = 8;
	int size_Z = 8;
	int size_N;
	int ctr = 0;

	Book* xlBook;
	Sheet* xlSheet;

	char* xlBookName_Single = "ex_fftwf_complexArray_Single.xls";
	char* xlBookName_Double= "ex_fftwf_complexArray_Double.xls";

	fftwf_complex* fftwfCmplxArray_1D;
	fftwf_complex* fftwfCmplxArray_2D_flat;
	fftwf_complex* fftwfCmplxArray_3D_flat;
	fftwf_complex** fftwfCmplxArray_2D;
	fftwf_complex*** fftwfCmplxArray_3D;

	fftw_complex* fftwCmplxArray_1D;
	fftw_complex* fftwCmplxArray_2D_flat;
	fftw_complex* fftwCmplxArray_3D_flat;
	fftw_complex** fftwCmplxArray_2D;
	fftw_complex*** fftwCmplxArray_3D;
}

void ex_fftwComplexArray::DumpNumbers_Single()
{
	INFO("ex_fftwf_ComplexArray - Single Precision");

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
			fftwfCmplxArray_1D = MEM_ALLOC_1D(fftwf_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fftwComplex::fillArray_1D(fftwfCmplxArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwfCmplxArray_1D[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwfCmplxArray_1D[j][1]);
			}

			// Filling array (Random)
			Array::fftwComplex::fillArray_1D(fftwfCmplxArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwfCmplxArray_1D[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwfCmplxArray_1D[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwfCmplxArray_1D);
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
			fftwfCmplxArray_2D_flat = MEM_ALLOC_1D(fftwf_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fftwComplex::fillArray_2D_flat(fftwfCmplxArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwfCmplxArray_2D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwfCmplxArray_2D_flat[j][1]);
			}

			// Filling array (Random)
			Array::fftwComplex::fillArray_2D_flat(fftwfCmplxArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwfCmplxArray_2D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwfCmplxArray_2D_flat[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwfCmplxArray_2D_flat);
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
			fftwfCmplxArray_3D_flat = MEM_ALLOC_1D(fftwf_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwComplex::fillArray_3D_flat(fftwfCmplxArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwfCmplxArray_3D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwfCmplxArray_3D_flat[j][1]);
			}

			// Filling array (Random)
			Array::fftwComplex::fillArray_3D_flat(fftwfCmplxArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwfCmplxArray_3D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwfCmplxArray_3D_flat[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwfCmplxArray_3D_flat);
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
			fftwfCmplxArray_2D = MEM_ALLOC_2D_FFTWFCOMPLEX(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwComplex::fillArray_2D(fftwfCmplxArray_2D, size_X, size_Y, 1);
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
					xlSheet->writeNum((ctr) + 2, 0, fftwfCmplxArray_2D[i][j][0]);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 1, fftwfCmplxArray_2D[i][j][1]);
				}

			// Filling array (Random)
			Array::fftwComplex::fillArray_2D(fftwfCmplxArray_2D, size_X, size_Y, 0);
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
					xlSheet->writeNum((ctr) + 2, 2, fftwfCmplxArray_2D[i][j][0]);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 3, fftwfCmplxArray_2D[i][j][1]);
				}

			// Freeing memory
			FREE_MEM_2D_FFTWFCOMPLEX(fftwfCmplxArray_2D, size_X, size_Y);
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
			fftwfCmplxArray_3D = MEM_ALLOC_3D_FFTWFCOMPLEX(size_X, size_Y, size_Z);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwComplex::fillArray_3D(fftwfCmplxArray_3D, size_X, size_Y, size_Z, 1);
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
						xlSheet->writeNum((ctr) + 2, 0, fftwfCmplxArray_3D[i][j][k][0]);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 1, fftwfCmplxArray_3D[i][j][k][1]);
					}

			// Filling array (Random)
			Array::fftwComplex::fillArray_3D(fftwfCmplxArray_3D, size_X, size_Y, size_Z, 0);
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
						xlSheet->writeNum((ctr) + 2, 2, fftwfCmplxArray_3D[i][j][k][0]);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 3, fftwfCmplxArray_3D[i][j][k][1]);
					}

			// Freeing memory
			FREE_MEM_3D_FFTWFCOMPLEX(fftwfCmplxArray_3D, size_X, size_Y, size_Z);
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

void ex_fftwComplexArray::DumpNumbers_Double()
{
	INFO("ex_fftwf_complexArray - Double Precision");

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
			fftwCmplxArray_1D = MEM_ALLOC_1D(fftw_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fftwDoubleComplex::fillArray_1D(fftwCmplxArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwCmplxArray_1D[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwCmplxArray_1D[j][1]);
			}

			// Filling array (Random)
			Array::fftwDoubleComplex::fillArray_1D(fftwCmplxArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwCmplxArray_1D[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwCmplxArray_1D[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwCmplxArray_1D);
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
			fftwCmplxArray_2D_flat = MEM_ALLOC_1D(fftw_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fftwDoubleComplex::fillArray_2D_flat(fftwCmplxArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwCmplxArray_2D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwCmplxArray_2D_flat[j][1]);
			}

			// Filling array (Random)
			Array::fftwDoubleComplex::fillArray_2D_flat(fftwCmplxArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwCmplxArray_2D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwCmplxArray_2D_flat[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwCmplxArray_2D_flat);
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
			fftwCmplxArray_3D_flat = MEM_ALLOC_1D(fftw_complex, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwDoubleComplex::fillArray_3D_flat(fftwCmplxArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Re-Seq");
			xlSheet->writeStr(1, 1, "Im-Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 0, fftwCmplxArray_3D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 1, fftwCmplxArray_3D_flat[j][1]);
			}

			// Filling array (Random)
			Array::fftwDoubleComplex::fillArray_3D_flat(fftwCmplxArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 2, "Re-Rnd");
			xlSheet->writeStr(1, 3, "Im-Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
			{
				// Real component
				xlSheet->writeNum(j + 2, 2, fftwCmplxArray_3D_flat[j][0]);

				// Imaginary component
				xlSheet->writeNum(j + 2, 3, fftwCmplxArray_3D_flat[j][1]);
			}

			// Freeing memory
			FREE_MEM_1D(fftwCmplxArray_3D_flat);
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
			fftwCmplxArray_2D = MEM_ALLOC_2D_FFTWCOMPLEX(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwDoubleComplex::fillArray_2D(fftwCmplxArray_2D, size_X, size_Y, 1);
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
					xlSheet->writeNum((ctr) + 2, 0, fftwCmplxArray_2D[i][j][0]);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 1, fftwCmplxArray_2D[i][j][1]);
				}

			// Filling array (Random)
			Array::fftwDoubleComplex::fillArray_2D(fftwCmplxArray_2D, size_X, size_Y, 0);
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
					xlSheet->writeNum((ctr) + 2, 2, fftwCmplxArray_2D[i][j][0]);

					// Imaginary component
					xlSheet->writeNum((ctr++) + 2, 3, fftwCmplxArray_2D[i][j][1]);
				}

			// Freeing memory
			FREE_MEM_2D_FFTWCOMPLEX(fftwCmplxArray_2D, size_X, size_Y);
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
			fftwCmplxArray_3D = MEM_ALLOC_3D_FFTWCOMPLEX(size_X, size_Y, size_Z);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fftwDoubleComplex::fillArray_3D(fftwCmplxArray_3D, size_X, size_Y, size_Z, 1);
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
						xlSheet->writeNum((ctr) + 2, 0, fftwCmplxArray_3D[i][j][k][0]);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 1, fftwCmplxArray_3D[i][j][k][1]);
					}

			// Filling array (Random)
			Array::fftwDoubleComplex::fillArray_3D(fftwCmplxArray_3D, size_X, size_Y, size_Z, 0);
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
						xlSheet->writeNum((ctr) + 2, 2, fftwCmplxArray_3D[i][j][k][0]);

						// Imaginary component
						xlSheet->writeNum((ctr++) + 2, 3, fftwCmplxArray_3D[i][j][k][1]);
					}

			// Freeing memory
			FREE_MEM_3D_FFTWCOMPLEX(fftwCmplxArray_3D, size_X, size_Y, size_Z);
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
