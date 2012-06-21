/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#ifndef VOLUME_H_
#define VOLUME_H_


#include <X11/X.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <math.h>
#include <sys/timeb.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "Globals.h"
#include "Utilities/Utils.h"


namespace Volume
{
	void testVolume();
    //char* loadVolume( const char* volFileName, const size_t volFileSize );

    volume_char_t loadVolume(const char* volName, const char* volPath);

    void createVolume_float( const int volSize, char* charVol );

    char*** allocateVolume_3D_char(int volSize_X, int volSize_Y, int volSize_Z);
	float*** allocateVolume_3D_float(int volSize_X, int volSize_Y, int volSize_Z);

	char*** convertVolume_1D_To_3D_char(const char* inputVol,
			const int volSize_X, const int volSize_Y, const int volSize_Z);
	float*** convertVolume_1D_To_3D_float(const float* inputVol,
			const int volSize_X, const int volSize_Y, const int volSize_Z);

	char* convertVolume_3D_To_1D_char(const char*** inputVol,
			const int volSize_X, const int volSize_Y, const int volSize_Z);
	float* convertVolume_3D_To_1D_float(const float*** inputVol,
			const int volSize_X, const int volSize_Y, const int volSize_Z);

	char*** extractVolume_char(const char*** inputVol,
			const int volSize_X, const int volSize_Y, const int volSize_Z,
			const int start_X, const int start_Y, const int start_Z);

	float*** extractVolume_float(const float*** inputVol,
				const int volSize_X, const int volSize_Y, const int volSize_Z,
				const int start_X, const int start_Y, const int start_Z);

	int freeVolume_1D_char(char* charVol);
	int freeVolume_1D_float(float* floatVol);
	int freeVolume_3D_char(char*** charVol);
	int freeVolume_3D_float(float*** floatVol);


	volumeDimensions_t openVolHeader(const char* volName, const char* volPath);
	vol_char_t openVolumeFile(const char* volName, const char* volPath, volumeDimensions_t volDim);


    volumeDimensions_t openVolHeaderFile( const char* volHdrFile);



    volume_complex_float_t createComplexVolume_float(volume_float_t volReal);
    volume_complex_double_t createComplexVolume_double(volume_double_t volReal);
    volume_complex_float_t createComplexVolumeFromChar_float(volume_char_t volReal);
    volume_complex_double_t createComplexVolumeFromChar_double(volume_char_t volReal);
}

#endif /* VOLUME_H_ */
