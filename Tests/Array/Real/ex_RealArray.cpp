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

#include "ex_RealArray.h"
#include "Utilities/XL.h"
#include "Array/Real/Array.h"
#include <cufft.h>
#include "Utilities/Memory.h"
#include "Utilities/MACROS.h"
#include <iostream>

namespace ex_RealArray
{
	int size_X = 8;
	int size_Y = 8;
	int size_Z = 8;
	int size_N;
	int ctr = 0;

	Book* xlBook;
	Sheet* xlSheet;

	char* xlBookName_Single = "ex_Array_Single.xls";
	char* xlBookName_Double= "ex_Array_Double.xls";

	float* floatArray_1D;
	float* floatArray_2D_flat;
	float* floatArray_3D_flat;
	float** floatArray_2D;
	float*** floatArray_3D;

	double* doubleArray_1D;
	double* doubleArray_2D_flat;
	double* doubleArray_3D_flat;
	double** doubleArray_2D;
	double*** doubleArray_3D;
}

void ex_RealArray::DumpNumbers_Single()
{
	INFO("ex_RealArray - Single Precision");

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
			floatArray_1D = MEM_ALLOC_1D(float, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_1D_float(floatArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, floatArray_1D[j]);

			// Filling array (Random)
			Array::fillArray_1D_float(floatArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, floatArray_1D[j]);


			// Freeing memory
			FREE_MEM_1D(floatArray_1D);
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
			floatArray_2D_flat = MEM_ALLOC_1D(float, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_2D_flat_float(floatArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, floatArray_2D_flat[j]);

			// Filling array (Random)
			Array::fillArray_2D_flat_float(floatArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, floatArray_2D_flat[j]);

			// Freeing memory
			FREE_MEM_1D(floatArray_2D_flat);
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
			floatArray_3D_flat = MEM_ALLOC_1D(float, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fillArray_3D_flat_float(floatArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, floatArray_3D_flat[j]);

			// Filling array (Random)
			Array::fillArray_3D_flat_float(floatArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, floatArray_3D_flat[j]);

			// Freeing memory
			FREE_MEM_1D(floatArray_3D_flat);
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
			floatArray_2D = MEM_ALLOC_2D_FLOAT(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_2D_float(floatArray_2D, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, 0, floatArray_2D[i][j]);

			// Filling array (Random)
			Array::fillArray_2D_float(floatArray_2D, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, 1, floatArray_2D[i][j]);

			// Freeing memory
			FREE_MEM_2D_FLOAT(floatArray_2D, size_X, size_Y);
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
			floatArray_3D = MEM_ALLOC_3D_FLOAT(size_X, size_Y, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_3D_float(floatArray_3D, size_X, size_Y, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
						xlSheet->writeNum((ctr++) + 2, 0, floatArray_3D[i][j][k]);

			// Filling array (Random)
			Array::fillArray_3D_float(floatArray_3D, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
						xlSheet->writeNum((ctr++) + 2, 1, floatArray_3D[i][j][k]);

			// Freeing memory
			FREE_MEM_3D_FLOAT(floatArray_3D, size_X, size_Y, size_Z);
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

void ex_RealArray::DumpNumbers_Double()
{
	INFO("ex_RealArray - Double Precision");

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
			doubleArray_1D = MEM_ALLOC_1D(double, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_1D_double(doubleArray_1D, size_N, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, doubleArray_1D[j]);


			// Filling array (Random)
			Array::fillArray_1D_double(doubleArray_1D, size_N, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, doubleArray_1D[j]);

			// Freeing memory
			FREE_MEM_1D(doubleArray_1D);
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
			doubleArray_2D_flat = MEM_ALLOC_1D(double, size_N);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_2D_flat_double(doubleArray_2D_flat, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, doubleArray_2D_flat[j]);

			// Filling array (Random)
			Array::fillArray_2D_flat_double(doubleArray_2D_flat, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, doubleArray_2D_flat[j]);

			// Freeing memory
			FREE_MEM_1D(doubleArray_2D_flat);
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
			doubleArray_3D_flat = MEM_ALLOC_1D(double, size_N);
			INFO("Array allocation done");

			// Filling array (Sequenatil)
			Array::fillArray_3D_flat_double(doubleArray_3D_flat, size_X, size_Y, size_Z, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 0, doubleArray_3D_flat[j]);

			// Filling array (Random)
			Array::fillArray_3D_flat_double(doubleArray_3D_flat, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			for (int j = 0; j < size_N; j++)
				xlSheet->writeNum(j + 2, 1, doubleArray_3D_flat[j]);

			// Freeing memory
			FREE_MEM_1D(doubleArray_3D_flat);
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
			doubleArray_2D = MEM_ALLOC_2D_DOUBLE(size_X, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_2D_double(doubleArray_2D, size_X, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, 0, doubleArray_2D[i][j]);

			// Filling array (Random)
			Array::fillArray_2D_double(doubleArray_2D, size_X, size_Y, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					xlSheet->writeNum((ctr++) + 2, 1, doubleArray_2D[i][j]);

			// Freeing memory
			FREE_MEM_2D_DOUBLE(doubleArray_2D, size_X, size_Y);
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
			doubleArray_3D = MEM_ALLOC_3D_DOUBLE(size_X, size_Y, size_Y);
			INFO("Array allocation done");

			// Filling array (Sequential)
			Array::fillArray_3D_double(doubleArray_3D, size_X, size_Y, size_Y, 1);
			INFO("Filing array done - Sequential");

			// Headers
			xlSheet->writeStr(1, 0, "Seq");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
						xlSheet->writeNum((ctr++) + 2, 0, doubleArray_3D[i][j][k]);

			// Filling array (Random)
			Array::fillArray_3D_double(doubleArray_3D, size_X, size_Y, size_Z, 0);
			INFO("Filing array done - Random");

			// Headers
			xlSheet->writeStr(1, 1, "Rnd");

			// Filling column with data
			ctr = 0;
			for (int i = 0; i < size_X; i++)
				for (int j = 0; j < size_Y; j++)
					for (int k = 0; k < size_Z; k++)
						xlSheet->writeNum((ctr++) + 2, 1, doubleArray_3D[i][j][k]);

			// Freeing memory
			FREE_MEM_3D_DOUBLE(doubleArray_3D, size_X, size_Y, size_Z);
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
