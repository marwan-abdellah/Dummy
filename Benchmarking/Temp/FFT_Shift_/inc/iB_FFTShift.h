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

#include "Globals.h"
#include "Utilities/Utils.h"
#include "Utilities/MACROS.h"
#include "Array/Real/Array.h"
#include "Utilities/XL.h"
#include "FFT/FFTShift.h"
#include "CUDA/Utilities/cuUtils.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/cuExterns.h"

namespace iB_FFTShift
{
	void FFTShift_1D_CPU(int size_X);
	void FFTShift_1D_CUDA(int size_X);

	void FFTShift_2D_Float(int size_X, int size_Y, Sheet* xlSheet, int nLoop);
	void FFTShift_2D_CUDA(int size_X, int size_Y);

	void FFTShift_3D_CPU(int size_X, int size_Y, int size_Z);
	void FFTShift_3D_CUDA(int size_X, int size_Y, int size_Z);

}
