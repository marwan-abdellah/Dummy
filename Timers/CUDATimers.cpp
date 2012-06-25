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
#include "CUDATimers.h"
#include "Utilities/MACROS.h"

timer_cuda Timers::CUDATimers::initTimer()
{
	timer_cuda kernelTime;

	// Create the timer
	cutCreateTimer(&kernelTime);

	// Initialize the timer to ZEROs
	cutResetTimer(kernelTime);

	return kernelTime;
}

durationStruct Timers::CUDATimers::getDuration(timer_cuda cuTimer)
{
	/* Returns the time in milli-seconds */
	long valDuration = cutGetTimerValue(cuTimer);

	durationStruct duration;
	duration.unit_NanoSec = valDuration * 1000000;
	duration.unit_MicroSec = valDuration * 1000;
	duration.unit_MilliSec = valDuration;
	duration.unit_Sec = valDuration / 1000;

	return duration;
}



