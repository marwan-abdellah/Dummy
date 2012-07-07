

#ifndef _MAXSUBARRAY_KERNEL_CU_
#define _MAXSUBARRAY_KERNEL_CU_

#include "Shared.h"
#include <cutil_inline.h>


__global__ 
void findMax( int rows, int cols, Max* dev_maxValues, int* dev_inputArray ) 
{

	int g = threadIdx.x;
	int tempMaxSum = 0;
	int candMaxSubArr = 0;
	int j = 0;
	int pr [1024]; 

	for( int i = g; i < rows; i++)
	{
		tempMaxSum = 0;
		j = 0;
		for(int h = 0; h < cols; h++)
		{
			pr[h] = pr[h] + dev_inputArray[i * cols + h];
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
