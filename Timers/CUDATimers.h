/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : ex_BoostTimer.h
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#ifndef CUDATIMERS_H_
#define CUDATIMERS_H_

#include "TimerGlobals.h"
#include "CUDA/cuGlobals.h"

namespace Timers
{
	namespace CUDATimers
	{
		timer_cuda initTimer();
		durationStruct getDuration(timer_cuda cuTimer);
	}

}
#endif /* CUDATIMERS_H_ */
