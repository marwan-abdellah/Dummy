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

#ifndef CUCOMPLEX_H_
#define CUCOMPLEX_H_

#include <cufft.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include "Utilities/Utils.h"

namespace Array
{
	namespace cuComplex
	{
		void zeroArray_1D(cufftComplex* arr, int size_X);
		void zeroArray_2D_flat(cufftComplex* arr, int size_X, int size_Y);
		void zeroArray_3D_flat(cufftComplex* arr, int size_X, int size_Y, int size_Z);
		void zeroArray_2D(cufftComplex** arr, int size_X, int size_Y);
		void zeroArray_3D(cufftComplex*** arr, int size_X, int size_Y, int size_Z);

		void fillArray_1D(cufftComplex* arr, int size_X, bool Seq_Rnd);
		void fillArray_2D_flat(cufftComplex* arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D_flat(cufftComplex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
		void fillArray_2D(cufftComplex** arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D(cufftComplex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	}

	namespace cuDoubleComplex
	{
		void zeroArray_1D(cufftDoubleComplex* arr, int size_X);
		void zeroArray_2D_flat(cufftDoubleComplex* arr, int size_X, int size_Y);
		void zeroArray_3D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, int size_Z);
		void zeroArray_2D(cufftDoubleComplex** arr, int size_X, int size_Y);
		void zeroArray_3D(cufftDoubleComplex*** arr, int size_X, int size_Y, int size_Z);

		void fillArray_1D(cufftDoubleComplex* arr, int size_X, bool Seq_Rnd);
		void fillArray_2D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D_flat(cufftDoubleComplex* arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
		void fillArray_2D(cufftDoubleComplex** arr, int size_X, int size_Y, bool Seq_Rnd);
		void fillArray_3D(cufftDoubleComplex*** arr, int size_X, int size_Y, int size_Z, bool Seq_Rnd);
	}
}

#endif /* CUCOMPLEX_H_ */
