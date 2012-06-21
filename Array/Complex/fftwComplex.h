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

#ifndef FFTWCOMPLEX_H_
#define FFTWCOMPLEX_H_

#include <cufft.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include "Utilities/Utils.h"

namespace Array
{
	namespace fftwComplex
	{
		void fillArray_1D(fftwf_complex* arr, int size_X, bool Seq_Rnd);
		void fillArray_2D_flat(fftwf_complex* arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D_flat(fftwf_complex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
		void fillArray_2D(fftwf_complex** arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D(fftwf_complex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	}

	namespace fftwDoubleComplex
	{
		void fillArray_1D(fftw_complex* arr, int size_X, bool Seq_Rnd);
		void fillArray_2D_flat(fftw_complex* arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D_flat(fftw_complex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
		void fillArray_2D(fftw_complex** arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D(fftw_complex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	}
}

#endif /* FFTWCOMPLEX_H_ */
