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

#include "ex_MaxSubArray.h"

int main(int argc, char** argv)
{
	int* arr = (int*)malloc(sizeof(int)*ex_MaxSubArray::rows * ex_MaxSubArray::cols);
	int cores = atoi (argv[1]);
	for(int a=2;a<argc;a++)
	{
		char* fname=argv[a];
		ex_MaxSubArray::readFile(fname, arr);
		//for(int i=0;i<ex_MaxSubArray::rows;i++)
		//	for(int j=0;j<ex_MaxSubArray::cols;j++)
		//		cout<<arr[i*ex_MaxSubArray::rows+j]<<endl;
		ex_MaxSubArray::getMax_CPU(arr, cores);
		
		/*
		// allocate an array to hold the maximum of all possible combination
		Max host_maxValues[ex_MaxSubArray::rows];

		ex_MaxSubArray::MaxSubArray_CUDA(arr, host_maxValues);

		int S = 0,ind=0;
		// search for the maximum value in all maximum candidates
		for (int i = 0; i < ex_MaxSubArray::rows; i++)
		{
			if (host_maxValues[i].S >S)
			{
				S = host_maxValues[i].S;
				ind=i;
			}
		}

		cout << host_maxValues[ind].y1 << " " << host_maxValues[ind].x1 << " " << host_maxValues[ind].y2 << " "  
			<< host_maxValues[ind].x2 <<" "<< endl;
			*/
	}

	free(arr);
	return 0;
}
