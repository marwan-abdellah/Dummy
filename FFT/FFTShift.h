/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Typedefs.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#ifndef FFT_SHIFT_H_
#define FFT_SHIFT_H_

#include "Globals.h"
#include "CUDA/cuGlobals.h"
#include "Utilities/MACROS.h"
#include "Timers/BoostTimers.h"
#include "Timers/TimerGlobals.h"

namespace FFT
{
	/* @ float*/
	float* FFT_Shift_1D_float(float* input, int nX, durationStruct* duration);
	float** FFT_Shift_2D_float(float** input, int nX, int nY, durationStruct* duration);
	float*** FFT_Shift_3D_float(float*** input, int nX, int nY, int nZ, durationStruct* duration);

	float* repack_2D_float(float** input_2D, int nX, int nY, durationStruct* duration);
	float* repack_3D_float(float*** input_3D, int nX, int nY, int nZ, durationStruct* duration);

	/* @ double */
	double* FFT_Shift_1D_double(double* input, int nX, durationStruct* duration);
	double** FFT_Shift_2D_double(double** input, int nX, int nY, durationStruct* duration);
	double*** FFT_Shift_3D_double(double*** input, int nX, int nY, int nZ, durationStruct* duration);

	double* repack_2D_double(double** input_2D, int nX, int nY, durationStruct* duration);
	double* repack_3D_double(double*** input_3D, int nX, int nY, int nZ, durationStruct* duration);

	/* @ cuComplex */
	cuComplex* FFT_Shift_1D_cuComplex(cuComplex* input, int nX, durationStruct* duration);
	cuComplex** FFT_Shift_2D_cuComplex(cuComplex** input, int nX, int nY, durationStruct* duration);
	cuComplex*** FFT_Shift_3D_cuComplex(cuComplex*** input, int nX, int nY, int nZ, durationStruct* duration);

	cuComplex* repack_2D_cuComplex(cuComplex** input_2D, int nX, int nY, durationStruct* duration);
	cuComplex* repack_3D_cuComplex(cuComplex*** input_3D, int nX, int nY, int nZ, durationStruct* duration);

	/* @ cuDoubleComplex */
	cuDoubleComplex* FFT_Shift_1D_cuDoubleComplex(cuDoubleComplex* input, int nX, durationStruct* duration);
	cuDoubleComplex** FFT_Shift_2D_cuDoubleComplex(cuDoubleComplex** input, int nX, int nY, durationStruct* duration);
	cuDoubleComplex*** FFT_Shift_3D_cuDoubleComplex(cuDoubleComplex*** input, int nX, int nY, int nZ, durationStruct* duration);

	cuDoubleComplex* repack_2D_cuDoubleComplex(cuDoubleComplex** input_2D, int nX, int nY, durationStruct* duration);
	cuDoubleComplex* repack_3D_cuDoubleComplex(cuDoubleComplex*** input_3D, int nX, int nY, int nZ, durationStruct* duration);
}


#endif /* FFT_SHIFT_H_ */
