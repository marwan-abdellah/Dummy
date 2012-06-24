/*
 * Globals.h
 *
 *  Created on: May 19, 2012
 *      Author: abdellah
 */

#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <fftw3.h>
#include "stlsoft/stlsoft.h"

/*
 * Gets rid of the annoying warrning
 * "deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]"
 */
#pragma GCC diagnostic ignored "-Wwrite-strings"

/* @ Max float for the rand() function */
#define FLOAT_MAX 214748364

/* @ Max GPU  Memory in MB */
#define MAX_GPU_MEMORY_MB 256
#define MAX_GPU_MEMORY MAX_GPU_MEMORY_MB * 1024 * 1024

struct volumeDimensions
{
    int size_X;
    int size_Y;
    int size_Z;
};

typedef volumeDimensions* volumeDimensions_t;
typedef char vol_char;
typedef float vol_float;
typedef double vol_double;
typedef vol_char* vol_char_t;
typedef vol_char* vol_float_t;
typedef vol_char* vol_double_t;

typedef fftwf_complex* fftwf_complex_t;
typedef fftw_complex* fftw_complex_t;


struct volume_char
{
	volumeDimensions_t volDim;
	vol_char_t volImg;
};

struct volume_float
{
	volumeDimensions_t volDim;
	vol_float_t volImg;
};

struct volume_double
{
	volumeDimensions_t volDim;
	vol_double_t volImg;
};

struct volume_complex_float
{
	volumeDimensions_t volDim;
	fftwf_complex_t volImg;
};

struct volume_complex_double
{
	volumeDimensions_t volDim;
	fftw_complex_t volImg;
};

typedef volume_char* volume_char_t;
typedef volume_float* volume_float_t;
typedef volume_double* volume_double_t;
typedef volume_complex_float* volume_complex_float_t;
typedef volume_complex_double* volume_complex_double_t;

struct profile
{
	double sec;
	double millsec;
	double microsec;
};

typedef profile stlTimer;

#endif /* _GLOBALS_H_ */
