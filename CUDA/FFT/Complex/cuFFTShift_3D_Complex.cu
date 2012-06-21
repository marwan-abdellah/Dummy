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

#ifndef CU_FFTSHIFT_3D_COMPLEX_CU_
#define CU_FFTSHIFT_3D_COMPLEX_CU_

#include <cutil_inline.h>

__global__
void fftShift_3D_Complex_i(cufftComplex* devArrayInput, cufftComplex* devArrayOutput, int arrSize1D, int zIndex)
{
    // 3D Volume & 2D Slice & 1D Line 
    int sLine = arrSize1D; 
    int sSlice = arrSize1D * arrSize1D; 
    int sVolume = arrSize1D * arrSize1D * arrSize1D; 
    
    // Transformations Equations 
    int sEq1 = (sVolume + sSlice + sLine) / 2;
    int sEq2 = (sVolume + sSlice - sLine) / 2; 
    int sEq3 = (sVolume - sSlice + sLine) / 2;
    int sEq4 = (sVolume - sSlice - sLine) / 2; 
    
    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index 2D  
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    // zIndex  
    
    // Thread Index Converted into 1D Index
    int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;
    
    if (zIndex < arrSize1D / 2)
    {
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
                devArrayOutput[index] = devArrayInput[index + sEq3]; 
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
                devArrayOutput[index] = devArrayInput[index + sEq4]; 
            }
        }
    }
    else 
    {
        if (xIndex < arrSize1D / 2)
        {
            if (yIndex < arrSize1D / 2)
            {
                // First Quad 
                devArrayOutput[index] = devArrayInput[index - sEq4]; 
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
                devArrayOutput[index] = devArrayInput[index - sEq3];
            }
            else 
            {
                // Fourth Quad
                devArrayOutput[index] = devArrayInput[index - sEq1]; 
            }
        }
    }
}

void fftShift_3D_Complex_i(cufftComplex* _arrayDeviceInput, cufftComplex* _arrayDeviceOutput, int _arraySize1D, dim3 _block, dim3 _grid)
{
    // Invoke Kernel 
    for (int i = 0; i < _arraySize1D; i++)
        fftShift_3D_Complex_i <<< _grid, _block >>> (_arrayDeviceInput, _arrayDeviceOutput, _arraySize1D, i);
}

__global__
void fftShift_3D_Double_Complex_i(cufftDoubleComplex* devArrayInput, cufftDoubleComplex* devArrayOutput, int arrSize1D, int zIndex)
{
    // 3D Volume & 2D Slice & 1D Line 
    int sLine = arrSize1D; 
    int sSlice = arrSize1D * arrSize1D; 
    int sVolume = arrSize1D * arrSize1D * arrSize1D; 
    
    // Transformations Equations 
    int sEq1 = (sVolume + sSlice + sLine) / 2;
    int sEq2 = (sVolume + sSlice - sLine) / 2; 
    int sEq3 = (sVolume - sSlice + sLine) / 2;
    int sEq4 = (sVolume - sSlice - sLine) / 2; 
    
    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index 2D  
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;
    // zIndex  
    
    // Thread Index Converted into 1D Index
    int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;
    
    if (zIndex < arrSize1D / 2)
    {
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
                devArrayOutput[index] = devArrayInput[index + sEq3]; 
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
                devArrayOutput[index] = devArrayInput[index + sEq4]; 
            }
        }
    }
    else 
    {
        if (xIndex < arrSize1D / 2)
        {
            if (yIndex < arrSize1D / 2)
            {
                // First Quad 
                devArrayOutput[index] = devArrayInput[index - sEq4]; 
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
                devArrayOutput[index] = devArrayInput[index - sEq3];
            }
            else 
            {
                // Fourth Quad
                devArrayOutput[index] = devArrayInput[index - sEq1]; 
            }
        }
    }
}

void fftShift_3D_Double_Complex_i(cufftDoubleComplex* _arrayDeviceInput, cufftDoubleComplex* _arrayDeviceOutput, int _arraySize1D, dim3 _block, dim3 _grid)
{
    // Invoke Kernel 
    for (int i = 0; i < _arraySize1D; i++)
        fftShift_3D_Double_Complex_i <<< _grid, _block >>> (_arrayDeviceInput, _arrayDeviceOutput, _arraySize1D, i);
}

#endif // CU_FFTSHIFT_3D_COMPLEX_CU_
