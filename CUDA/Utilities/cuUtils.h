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

#ifndef CUUTILS_H_
#define CUUTILS_H_

#include "CUDA/cuGlobals.h"
#include "Utilities/Utils.h"
#include "Globals.h"
#include "Utilities/Logging.h"
#include "Utilities/MACROS.h"

namespace cuUtils
{
	int upload_1D_float(float* hostArr, float* devArr, int size_X);
	int upload_2D_float(float* hostArr, float* devArr, int size_X, int size_Y);
	int upload_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_float(float* hostArr, float* devArr, int size_X);
	int download_2D_float(float* hostArr, float* devArr, int size_X, int size_Y);
	int download_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_double(double* hostArr, double* devArr, int size_X);
	int upload_2D_double(double* hostArr, double* devArr, int size_X, int size_Y);
	int upload_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_double(double* hostArr, double* devArr, int size_X);
	int download_2D_double(double* hostArr, double* devArr, int size_X, int size_Y);
	int download_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X);
	int upload_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y);
	int upload_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X);
	int download_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y);
	int download_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X);
	int upload_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y);
	int upload_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X);
	int download_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y);
	int download_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z);
}

#endif /* CUUTILS_H_ */
