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

#ifndef CU_FFTSHIFT_2D_REAL_CU_
#define CU_FFTSHIFT_2D_REAL_CU_

#include <cutil_inline.h>

__global__
void fftShift_2D_Kernel(float* devArrayOutput, float* devArrayInput, int arrSize1D)
{
    // 2D Slice & 1D Line 
    int sLine = arrSize1D; 
    int sSlice = arrSize1D * arrSize1D; 
    
    // Transformations Equations 
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2; 
    
    __syncthreads();
    
    // Thread Index (1D)
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;
    
    __syncthreads();

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    
    __syncthreads();

    // Thread Index (2D)  
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    
    __syncthreads();
    
    // Thread Index Converted into 1D Index
    int index = (yIndex * arrSize1D) + xIndex;
    
    __syncthreads();
    
    if (xIndex < arrSize1D / 2)
    {
        if (yIndex < arrSize1D / 2)
        {
            // First Quad 
            devArrayOutput[index] = devArrayInput[index + sEq1]; 
            __syncthreads();
        }
        else 
        {
            // Third Quad 
            devArrayOutput[index] = devArrayInput[index - sEq2];
            __syncthreads(); 
        }
    }
    else 
    {
        if (yIndex < arrSize1D / 2)
        {
            // Second Quad 
            devArrayOutput[index] = devArrayInput[index + sEq2];
            __syncthreads();
        }
        else 
        {
            // Fourth Quad
            devArrayOutput[index] = devArrayInput[index - sEq1]; 
            __syncthreads();
        }
    }
}

__global__
void fftShift_2D_Double_Kernel(double* devArrayOutput, double* devArrayInput, int arrSize1D)
{
    // 2D Slice & 1D Line 
    int sLine = arrSize1D; 
    int sSlice = arrSize1D * arrSize1D; 
    
    // Transformations Equations 
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2; 
    
    // Thread Index (1D)
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index (2D)  
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    
    // Thread Index Converted into 1D Index
    int index = (yIndex * arrSize1D) + xIndex;
    
    if (xIndex < arrSize1D / 2)
    {
        if (yIndex < arrSize1D / 2)
        {
            // First Quad 
            devArrayOutput[index] = devArrayInput[index + sEq1]; 
        }
        else 
        {
            // Third Quad 
            devArrayOutput[index] = devArrayInput[index - sEq2]; 
        }
    }
    else 
    {
        if (yIndex < arrSize1D / 2)
        {
            // Second Quad 
            devArrayOutput[index] = devArrayInput[index + sEq2];
        }
        else 
        {
            // Fourth Quad
            devArrayOutput[index] = devArrayInput[index - sEq1]; 
        }
    }
}

#endif // CU_FFTSHIFT_2D_REAL_CU_
