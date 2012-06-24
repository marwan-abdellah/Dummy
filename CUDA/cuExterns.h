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

#ifndef CUEXTERNS_H_
#define CUEXTERNS_H_

#include "cuGlobals.h"

extern
void cuCopyArray( dim3 cuBlock, dim3 cuGrid, float* devArrayOutput, float* devArrayInput,
	 			  int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_2D( dim3 cuBlock, dim3 cuGrid, float* devArrayOutput, float* devArrayInput,
					int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_2D_Double( dim3 cuBlock, dim3 cuGrid, double* devArrayOutput, double* devArrayInput,
						   int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_2D_Complex( dim3 cuBlock, dim3 cuGrid, cufftComplex* devArrayOutput, cufftComplex* devArrayInput,
							int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_2D_Double_Complex( dim3 cuBlock, dim3 cuGrid, cufftDoubleComplex* devArrayOutput, cufftDoubleComplex* devArrayInput,
								   int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_3D( dim3 cuBlock, dim3 cuGrid, float* devArrayOutput, float* devArrayInput,
					int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_3D_Double( dim3 cuBlock, dim3 cuGrid, double* devArrayOutput, double* devArrayInput,
						   int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_3D_Complex( dim3 cuBlock, dim3 cuGrid, cufftComplex* devArrayOutput, cufftComplex* devArrayInput,
						 	int nX, cudaProfile* cuProfile);

extern
void cuFFTShift_3D_Double_Complex( dim3 cuBlock, dim3 cuGrid, cufftDoubleComplex* devArrayOutput, cufftDoubleComplex* devArrayInput,
								   int nX, cudaProfile* cuProfile);


#endif /* CUEXTERNS_H_ */
