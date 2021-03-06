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
#include "Iterations.h"

namespace iB_Real_FFTShift_3D
{
	void FFTShift_3D_Float_Seq(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock);
	void FFTShift_3D_Double_Seq(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock);
	void FFTShift_3D_Float_Rnd(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock);
	void FFTShift_3D_Double_Rnd(int size_X, int size_Y, int size_Z, Sheet* xlSheet, int nLoop, dim3 cuGrid, dim3 cuBlock);
}
