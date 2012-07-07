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
 
#ifndef CU_EXTERNS_CU_
#define CU_EXTERNS_CU_

#include "CUDA/cuGlobals.h"

#include <cutil_inline.h>

#include "MaxSubArray.cu"
#include "Timers/BoostTimers.h"
#include "Shared.h"

extern  
void cuGetMax(dim3 cuBlock, dim3 cuGrid, Max* dev_maxValues, int* devArrayInput, int rows, int cols)
{
    findMax <<<cuBlock, cuGrid>>> (  rows,  cols, dev_maxValues, devArrayInput ); 
} 

#endif // CU_EXTERNS_CU_