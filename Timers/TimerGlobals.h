/*
 * TimerGlobals.h
 *
 *  Created on: Jun 20, 2012
 *      Author: abdellah
 */

#ifndef TIMERGLOBALS_H_
#define TIMERGLOBALS_H_

/* @ Boost timers */
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

/* @ Boost timer typedefs */
typedef boost::posix_time::ptime time_boost;
typedef boost::posix_time::time_duration duration_boost;

/* @ CUDA timer typedef */
typedef uint timer_cuda;

struct durationStruct
{
	double unit_NanoSec;
	double unit_MicroSec;
	double unit_MilliSec;
	double unit_Sec;
};


#endif /* TIMERGLOBALS_H_ */
