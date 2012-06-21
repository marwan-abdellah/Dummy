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

#include "FFT3D.h"

fftwf_complex_t FFT::forward_FFT_3D_float
	(fftwf_complex_t input, const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	// Allocate the output array
	fftwf_complex_t output = MEM_ALLOC_1D(fftwf_complex, (size_X * size_Y * size_Z));

	// Allocate the FFT execution plan
	fftwf_plan plan =
			fftwf_plan_dft_3d(size_X, size_Y, size_Z, input, output, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftwf_execute(plan);

	return output;
}

fftw_complex_t FFT::forward_FFT_3D_double
	(fftw_complex_t input, const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	// Allocate the output array
	fftw_complex_t output = MEM_ALLOC_1D(fftw_complex, (size_X * size_Y * size_Z));

	// Allocate the FFT execution plan
	fftw_plan plan = fftw_plan_dft_3d
			(size_X, size_Y, size_Z, input, output, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftw_execute(plan);

	return output;
}

fftwf_complex_t FFT::inverse_FFT_3D_float
	(fftwf_complex_t input, const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	// Allocate the output array
	fftwf_complex_t output = MEM_ALLOC_1D(fftwf_complex, (size_X * size_Y * size_Z));

	// Allocate the FFT execution plan
	fftwf_plan plan =
			fftwf_plan_dft_3d(size_X, size_Y, size_Z, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftwf_execute(plan);

	return output;
}

fftw_complex_t FFT::inverse_FFT_3D_doube
	(fftw_complex_t input, const int size_X, const int size_Y, const int size_Z)
{
	LOG();

	// Allocate the output array
	fftw_complex_t output = MEM_ALLOC_1D(fftw_complex, (size_X * size_Y * size_Z));

	// Allocate the FFT execution plan
	fftw_plan plan =
			fftw_plan_dft_3d(size_X, size_Y, size_Z, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the plan
	fftw_execute(plan);

	return output;
}
