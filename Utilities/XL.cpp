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
#include "XL.h"

Book* Utils::xl::createBook()
{
	Book* xlBook = xlCreateBook();
	return xlBook;
}

Sheet* Utils::xl::addSheetToBook(char* sheetName, Book* xlBook)
{
	Sheet* xlSheet = xlBook->addSheet(sheetName);
	return xlSheet;
}

void Utils::xl::fillSheetWithSampleData(Sheet* xlSheet, int rowMax, int colMax)
{
	for (int j = 0; j < rowMax; j++)
		xlSheet->writeStr(1, j, "Header");

	// Data
	for (int i  = 0; i < colMax; i++)
		for (int j = 0; j < rowMax; j++)
			xlSheet->writeNum(i + 2, j, i);
}

void Utils::xl::saveBook(Book* xlBook, char* xlBookName)
{
	xlBook->save(xlBookName);
}

void Utils::xl::releaseBook(Book* xlBook)
{
	xlBook->release();
}

void Utils::Test_XL()
{
	// Create a xlCreateXMLBook() for ".xlsx"
	Book* xlBook = xlCreateBook();

	if(xlBook)
	{
		Sheet* xlSheet = xlBook->addSheet("TestSheet");
		if(xlSheet)
		{
			/* Start writing to the file from the second row
			 * if you don't have the full license.
			 */

			// Headers (Strings)
			for (int j = 0; j < 10; j++)
				xlSheet->writeStr(1, j, "Header");

			// Data
			for (int i  = 0; i < 10; i++)
				for (int j = 0; j < 10; j++)
					xlSheet->writeNum(i + 2, j, i);
		}

		// Save the XL file
		xlBook->save("TestXL.xls");

		// Release the file to be able to OPEN it
		xlBook->release();
	}
}
