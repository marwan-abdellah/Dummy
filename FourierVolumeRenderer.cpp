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

#include "FourierVolumeRenderer.h"
#include "Utilities/LoggingMACROS.h"
#include "Utilities/Memory.h"
#include <fftw3.h>
#include "FFT/FFT3D.h"

namespace FVR
{
	volumeDimensions_t _volDim;
	vol_char_t _volImage;

	char _volumeName[1000];
	char _volumeFileDir[1000];
	int _volSize_XYZ;

	volume_char_t _volReal_char;
	volume_float_t _volReal_float;
	volume_complex_float_t _volComplex;

}


void FVR::initApp()
{

}

void FVR::loadVolume()
{

}

void FVR::createSpectrum()
{

}

void FVR::initRenderingContext()
{

}

void FVR::uploadSpectrum()
{

}

void FVR::extractSlice()
{

}

void FVR::resampleSlice()
{

}

void FVR::getProjection()
{

}

void FVR::displayProjection()
{

}

void FVR::preProcessingStage()
{

}

void FVR::renderingLoop()
{

}

void FVR::freeData()
{

}

void FVR::doItAllHere()
{
	LOG();
	strcpy(_volumeName , "BONSAI");
	strcpy(_volumeFileDir , "/home/abdellah/Software/DataSets/BONSAI");

	//
	_volDim = MEM_ALLOC_1D(volumeDimensions, 1);

	// Loading the volume dimensions from the volume header file
	_volDim =  Volume::openVolHeader(_volumeName, _volumeFileDir);
	_volSize_XYZ = (_volDim->size_X * _volDim->size_Y * _volDim->size_Z);

	//
	_volImage = MEM_ALLOC_1D(vol_char, _volSize_XYZ);

	// Loading the volume image
	_volImage = Volume::openVolumeFile(_volumeName, _volumeFileDir, _volDim);

	_volReal_char = MEM_ALLOC_1D(volume_char, 1);
	_volReal_char->volImg = _volImage;
	_volReal_char->volDim = _volDim;

	// Create fftw_complex array and deleting the char one
	_volComplex = MEM_ALLOC_1D(volume_complex_float, 1);

	// Creae complex volume
	_volComplex = Volume::createComplexVolumeFromChar_float(_volReal_char);

	// FREE
	FREE_MEM_1D(_volReal_char->volImg);
	FREE_MEM_1D(_volReal_char->volDim);
	FREE_MEM_1D(_volReal_char);

	// DO the 3D FFT
	_volComplex->volImg = FFT::forward_FFT_3D_float(_volComplex->volImg,
											_volComplex->volDim->size_X,
											_volComplex->volDim->size_Y,
											_volComplex->volDim->size_Z);

	// Init opengl context

	//




































}
