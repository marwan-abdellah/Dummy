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

#ifndef CUGLOBALS_H_
#define CUGLOBALS_H_

#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct profileStruct
{
	uint kernelTime;
	float kernelDuration;
	int kernelExecErr;
};

typedef profileStruct cudaProfile;

#endif /* CUGLOBALS_H_ */
