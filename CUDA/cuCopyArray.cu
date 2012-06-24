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



#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void copyArray_2D_float_kernel(float* devArrayOutput, float* devArrayInput, int nX)
{
 
    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index 2D  
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    
    int index = (yIndex * nX) + xIndex;
    devArrayOutput[index] =( nX * nX )- index; 
}
    

    