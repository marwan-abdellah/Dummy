/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : ex_BoostTimer.cpp
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#include "ex_BoostTimer.h"
#include "Array/Real/Array.h"
#include "Utilities/MACROS.h"
#include <iostream>
#include "Timers/BoostTimers.h"

namespace ex_BoostTimer
{
	/* @ To be used for command line example extension */
	int stepsMilliSec = 0;
	int iterationTimer = 10;

	time_boost start;
	time_boost end;
	durationStruct* duration;
	durationStruct* durationTotal;
}

void ex_BoostTimer::ClearDurationCounters(durationStruct* durationCounter)
{
	durationCounter->unit_NanoSec = 0;
	durationCounter->unit_MicroSec = 0;
	durationCounter->unit_MilliSec = 0;
	durationCounter->unit_Sec = 0;
}

void ex_BoostTimer::TestTimersReslution()
{
	INFO("ex_BoostTimers - Single Timer");

	// Allocating Timers
	duration = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

	/**********************************************************
	 * 5 micro seconds Case
	 **********************************************************/
	INFO("5 micro seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::microsec(5));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 50 micro seconds Case
	 **********************************************************/
	INFO("50 micro seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::microsec(50));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 500 micro seconds Case
	 **********************************************************/
	INFO("500 micro seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::microsec(500));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 5 milli seconds Case
	 **********************************************************/
	INFO("5 milli seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::millisec(5));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 50 milli seconds Case
	 **********************************************************/
	INFO("50 milli seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::millisec(50));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 500 milli seconds Case
	 **********************************************************/
	INFO("500 milli seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::millisec(500));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));
	SEP();

	/**********************************************************
	 * 5 seconds Case
	 **********************************************************/
	INFO("5 seconds Case");
	start = Timers::BoostTimers::getTime_MicroSecond();
	boost::this_thread::sleep(boost::posix_time::seconds(5));
	end = Timers::BoostTimers::getTime_MicroSecond();
	duration = Timers::BoostTimers::getDuration(start, end);

	INFO("NANO Seconds: " + ITS(duration->unit_NanoSec));
	INFO("MICRO Seconds: " + ITS(duration->unit_MicroSec));
	INFO("MILLI Seconds: " + ITS(duration->unit_MilliSec));
	INFO("Seconds: " + ITS(duration->unit_Sec));

	// Freeing memory
	FREE_MEM_1D(duration);
}

void ex_BoostTimer::TestTimersReslutionWithAverage()
{
	INFO("ex_BoostTimers - Timer Average");

	// Allocating Timers
	duration = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
	durationTotal = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

	/**********************************************************
	 * 5 micro seconds Case
	 **********************************************************/
	INFO("5 micro seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::microsec(5));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();

	/**********************************************************
	 * 50 micro seconds Case
	 **********************************************************/
	INFO("50 micro seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::microsec(50));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();


	/**********************************************************
	 * 500 micro seconds Case
	 **********************************************************/
	INFO("500 micro seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::microsec(500));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();

	/**********************************************************
	 * 5 milli seconds Case
	 **********************************************************/
	INFO("5 milli seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::millisec(5));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();

	/**********************************************************
	 * 50 milli seconds Case
	 **********************************************************/
	INFO("50 milli seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::millisec(50));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();

	/**********************************************************
	 * 500 milli seconds Case
	 **********************************************************/
	INFO("500 milli seconds Case");

	ClearDurationCounters(durationTotal);

	for (int i = 0; i < iterationTimer; i++)
	{
		start = Timers::BoostTimers::getTime_MicroSecond();
		boost::this_thread::sleep(boost::posix_time::millisec(500));
		end = Timers::BoostTimers::getTime_MicroSecond();
		duration = Timers::BoostTimers::getDuration(start, end);

		durationTotal->unit_NanoSec += duration->unit_NanoSec;
		durationTotal->unit_MicroSec += duration->unit_MicroSec;
		durationTotal->unit_MilliSec += duration->unit_MilliSec;
		durationTotal->unit_Sec += duration->unit_Sec;
	}

	INFO("NANO Seconds: " + ITS(durationTotal->unit_NanoSec / iterationTimer));
	INFO("MICRO Seconds: " + ITS(durationTotal->unit_MicroSec / iterationTimer));
	INFO("MILLI Seconds: " + ITS(durationTotal->unit_MilliSec / iterationTimer));
	INFO("Seconds: " + ITS(durationTotal->unit_Sec / iterationTimer));
	SEP();

	// Freeing memory
	FREE_MEM_1D(duration);
	FREE_MEM_1D(durationTotal);
}
