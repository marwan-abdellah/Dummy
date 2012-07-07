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
void findMax( int rows, int cols, Max* dev_maxValues, int* dev_inputArray ) 
{

	int g = blockIdx.x * blockDim.x + threadIdx.x;
	int tempMaxSum = 0;
    int candMaxSubArr = 0;
    int j = 0;
    int pr [1024]; 
    
    dev_maxValues[g].S = 0;
    for(int h = 0; h < cols; h++) 
          pr[h] = 0;

    for( int i = g; i < rows; i++)
    {
        tempMaxSum = 0;
        j = 0;
        for(int h = 0; h < cols; h++)
        {
            pr[h] = pr[h] + dev_inputArray[i * rows + h];
            tempMaxSum = tempMaxSum + pr[h]; // t is the prefix sum of the strip start at row z to row x

            if( tempMaxSum > candMaxSubArr)
            { 
                candMaxSubArr = tempMaxSum;
                dev_maxValues[g].S = candMaxSubArr;
                dev_maxValues[g].x1 = g;
                dev_maxValues[g].y1 = j;
                dev_maxValues[g].x2 = i;
                dev_maxValues[g].y2 = h;
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
