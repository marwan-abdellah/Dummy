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

#ifndef FOURIER_VOLUME_RENDERER_H_
#define FOURIER_VOLUME_RENDERER_H_

#include "Globals.h"
#include "Utilities/MACROS.h"
#include "Volume/Volume.h"

namespace FVR
{
	extern char volumeName[1000];
	extern char volumeFileDir[1000];

	extern volumeDimensions_t _volDim;


	void initApp();

	void loadVolume();

	void createSpectrum();

	void initRenderingContext();

	void uploadSpectrum();

	void extractSlice();

	void resampleSlice();

	void getProjection();

	void displayProjection();

	void preProcessingStage();

	void renderingLoop();

	void freeData();

	extern void doItAllHere();
}


#endif /* FOURIER_VOLUME_RENDERER_H_ */
