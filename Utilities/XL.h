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

#ifndef XL_H_
#define XL_H_

#include "libxl.h"
using namespace libxl;

namespace Utils
{
	namespace xl
	{
		Book* createBook();
		Sheet* addSheetToBook(char* sheetName, Book* xlBook);
		void fillSheetWithSampleData(Sheet* xlSheet, int rowMax, int colMax);
		void saveBook(Book* xlBook, char* xlBookName);
		void releaseBook(Book* xlBook);
	}
	void Test_XL();
}
#endif /* XL_H_ */
