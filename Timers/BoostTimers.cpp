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
#include "BoostTimers.h"
#include "Utilities/MACROS.h"

time_boost Timers::BoostTimers::getTime_Second()
{
	return (time_boost) boost::posix_time::second_clock::local_time();
}

time_boost Timers::BoostTimers::getTime_MicroSecond()
{
	return (time_boost) boost::posix_time::microsec_clock::local_time();
}

durationStruct* Timers::BoostTimers::getDuration(time_boost startTime, time_boost endTime)
{
	duration_boost durationCalc = endTime - startTime;

	durationStruct* duration = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

	duration->unit_NanoSec = (double) durationCalc.total_nanoseconds();
	duration->unit_MicroSec = (double) durationCalc.total_microseconds();
	duration->unit_MilliSec = (double) durationCalc.total_microseconds() / 1000;
	duration->unit_Sec = (double) durationCalc.total_microseconds()/ (1000 * 1000);

	return duration;
}





/*


*/
