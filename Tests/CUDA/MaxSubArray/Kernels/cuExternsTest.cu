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
 
#ifndef CU_EXTERNS_TEST_CU_
#define CU_EXTERNS_TEST_CU_

#include "CUDA/cuGlobals.h"

#include <cutil_inline.h>

#include "MaxSubArray.cu"
#include "Timers/BoostTimers.h"
#include "Shared.h"

extern  
void cuGetMax(dim3 cuBlock, dim3 cuGrid, 
              Max* dev_maxValues, int* devArrayInput, 
              int numRows, int numCols, cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    findMax <<<cuBlock, cuGrid>>> (numRows, numCols, dev_maxValues, devArrayInput); 
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
} 

extern  
void cuReduction(dim3 cuBlock, dim3 cuGrid, 
             Max* g_data, int blockSize, int& idxMax, 
             cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
   //reduction<<<cuBlock, cuGrid>>> (g_data, TileWidth) ; 
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
} 

#endif // CU_EXTERNS_TEST_CU_