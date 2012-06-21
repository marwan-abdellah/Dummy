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

#include "ex_LibXL.h"
#include "Utilities/XL.h"

namespace ex_LibXL
{
	Book* xlBook;
	Sheet* xlSheet_1;
	Sheet* xlSheet_2;

	char* xlBookName = "ex_LibXL_Book.xls";
	char* xlSheetName_1 = "ex_LibXL_Sheet_1";
	char* xlSheetName_2 = "ex_LibXL_Sheet_2";
}

void ex_LibXL::genSampleBook()
{
	INFO("ex_LibXL::genSampleBook()");

	// Create the XL book
	xlBook = Utils::xl::createBook();
	INFO("xlBook created");

	if(xlBook)
	{
		// Add 2 sheets to the XL book namely 1 & 2
		xlSheet_1 = Utils::xl::addSheetToBook(xlSheetName_1, xlBook);

		if (xlSheet_1)
			Utils::xl::fillSheetWithSampleData(xlSheet_1, 5, 5);
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}

		xlSheet_2 = Utils::xl::addSheetToBook(xlSheetName_2, xlBook);
		if (xlSheet_2)
			Utils::xl::fillSheetWithSampleData(xlSheet_2, 10, 10);
		else
		{
			INFO("No valid xlSheet was created, EXITTING ...");
			EXIT(0);
		}

		INFO("Sheets created & filled with DUMMY data");

		// Writhe the Xl book to disk
		Utils::xl::saveBook(xlBook, xlBookName);
		INFO("Saving xlBook to disk")

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
