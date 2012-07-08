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

#ifndef CUEXTERNS_TEST_H_
#define CUEXTERNS_TEST_H_

#include "CUDA/cuGlobals.h"

extern
void cuGetMax(dim3 cuBlock, dim3 cuGrid,
		Max* dev_maxValues, int* devArrayInput, int rows, int cols, cudaProfile* cuProfile);

extern
void cuReduction(dim3 cuBlock, dim3 cuGrid, Max* g_data, int TileWidth, cudaProfile* cuProfile);

#endif /* CUEXTERNS_TEST_H_ */
