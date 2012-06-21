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

#ifndef FFT2D_H_
#define FFT2D_H_

#include <fftw3.h>
#include "Globals.h"
#include "Utilities/MACROS.h"

namespace FFT
{
	fftwf_complex* forward_FFT_2D_float
	(fftwf_complex *input, const int size_X, const int size_Y);

	fftw_complex* forward_FFT_2D_double
	(fftw_complex* input, const int size_X, const int size_Y);

	fftwf_complex* inverse_FFT_2D_float
	(fftwf_complex* input, const int size_X, const int size_Y);

	fftw_complex* inverse_FFT_2D_doube
	(fftw_complex* input, const int size_X, const int size_Y);
}

#endif /* FFT2D_H_ */
