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

#ifndef _EX_MAX_SUB_ARRAY_KERNEL_H_
#define _EX_MAX_SUB_ARRAY_KERNEL_H_

#include "Globals.h"
#include "Utilities/Utils.h"
#include "Utilities/MACROS.h"
#include "Array/Real/Array.h"
#include "Utilities/XL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "Shared.h"

/* @ This struct will hold the maximum of all strips can be found for each row */


namespace ex_MaxSubArray
{
	void readFile(char* inputArray, int* numCores, int numRows, int numCols);
	void getMax_CPU(int* inputArray, int numCores, int numRows, int numCols);
	void getMax_CUDA(int* inputArray, Max* maxValue, int numRows, int numCols);
}

#endif // _EX_MAX_SUB_ARRAY_KERNEL_H_
