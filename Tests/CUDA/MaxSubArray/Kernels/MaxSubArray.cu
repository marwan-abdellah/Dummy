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

#ifndef _MAXSUBARRAY_KERNEL_CU_
#define _MAXSUBARRAY_KERNEL_CU_

#include <cutil_inline.h>
#include "Shared.h"

__global__ 
void findMax(int numRows, int numCols, Max* dev_maxValues, int* dev_inputArray) 
{   
    // Calculating the correct index from the configuration  
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// In between parameters 
	// TODO: Salah to indicate what are the parameters 
	int tempMaxSum = 0;
    int candMaxSubArr = 0;
    int j = 0;
    int prefSum [1024]; 
    
    dev_maxValues[index].val = 0;
    
    // Resetting the prefix sum array 
    for(int iCtr = 0; iCtr < numCols; iCtr++) 
        prefSum[iCtr] = 0;
    
    for(int i = index; i < numRows; i++)
    {
        tempMaxSum = 0;
        j = 0;
        
        for(int h = 0; h < numCols; h++)
        {
            prefSum[h] = prefSum[h] + dev_inputArray[i * numRows + h];
            tempMaxSum = tempMaxSum + prefSum[h]; // t is the prefix sum of the strip start at row z to row x

            if( tempMaxSum > candMaxSubArr)
            { 
                candMaxSubArr = tempMaxSum;
                dev_maxValues[index].val = candMaxSubArr;
                dev_maxValues[index].x1 = index;
                dev_maxValues[index].y1 = j;
                dev_maxValues[index].x2 = i;
                dev_maxValues[index].y2 = h;
            }
            
            if( tempMaxSum < 0 )
            {
                tempMaxSum = 0;
                j = h + 1;
            }
        }
    }
}

#endif  //_MAXSUBARRAY_KERNEL_CU_
