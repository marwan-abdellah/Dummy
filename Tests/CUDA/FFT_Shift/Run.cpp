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

#include "ex_FFTShift.h"

int main()
{
	//ex_FFTShift::FFTShift_2D_CPU(256, 256);


	ex_FFTShift::FFTShift_2D_CUDA(32, 32);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(64, 64);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(128, 128);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(256, 256);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(512, 512);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(1024, 1024);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(1024 * 2, 1024 * 2);
	SEP();
	ex_FFTShift::FFTShift_2D_CUDA(1024 * 4, 1024 * 4);

	//ex_FFTShift::FFTShift_3D_CPU(4, 4, 4);
	//ex_FFTShift::FFTShift_3D_CUDA(4,4, 4);

	return 0;
}
