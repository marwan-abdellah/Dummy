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

struct durationStruct
{
	long unit_NanoSec;
	long unit_MicroSec;
	long unit_MilliSec;
	long unit_Sec;
};


#endif /* TIMERGLOBALS_H_ */
