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

#include "FFT2D.h"

fftwf_complex* FFT::forward_FFT_2D_float
	(fftwf_complex* input, const int size_X, const int size_Y)
{
	LOG();

	// Allocate the output array
	fftwf_complex* output = MEM_ALLOC_1D(fftwf_complex, (size_X * size_Y));

	// Allocate the FFT execution plan
	fftwf_plan plan =
			fftwf_plan_dft_2d(size_X, size_Y, input, output, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftwf_execute(plan);

	return output;
}

fftw_complex* FFT::forward_FFT_2D_double
	(fftw_complex* input, const int size_X, const int size_Y)
{
	LOG();

	// Allocate the output array
	fftw_complex* output = MEM_ALLOC_1D(fftw_complex, (size_X * size_Y));

	// Allocate the FFT execution plan
	fftw_plan plan =
			fftw_plan_dft_2d(size_X, size_Y, input, output, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftw_execute(plan);

	return output;
}

fftwf_complex* FFT::inverse_FFT_2D_float
	(fftwf_complex* input, const int size_X, const int size_Y)
{
	LOG();

	// Allocate the output array
	fftwf_complex* output = MEM_ALLOC_1D(fftwf_complex, (size_X * size_Y));

	// Allocate the FFT execution plan
	fftwf_plan plan =
			fftwf_plan_dft_2d(size_X, size_Y, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftwf_execute(plan);

	return output;
}

fftw_complex* FFT::inverse_FFT_2D_doube
	(fftw_complex* input, const int size_X, const int size_Y)
{
	LOG();

	// Allocate the output array
	fftw_complex* output = MEM_ALLOC_1D(fftw_complex, (size_X * size_Y));

	// Allocate the FFT execution plan
	fftw_plan plan =
			fftw_plan_dft_2d(size_X, size_Y, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftw_execute(plan);

	return output;
}
