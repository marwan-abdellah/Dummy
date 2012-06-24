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

#ifndef BOOSTTIMERS_H_
#define BOOSTTIMERS_H_

#include "TimerGlobals.h"

namespace Timers
{
	namespace BoostTimers
	{
		time_boost getTime_MicroSecond();
		time_boost getTime_Second();
		durationStruct* getDuration(time_boost startTime, time_boost endTime);
	}

}
#endif /* BOOSTTIMERS_H_ */
