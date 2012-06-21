/*
 * Bench.h
 *
 *  Created on: Jun 17, 2012
 *      Author: abdellah
 */

#ifndef BENCH_H_
#define BENCH_H_

#include "Globals.h"
#include "Utilities/MACROS.h"

namespace Utils
{
	namespace Bench
	{
		namespace STL
		{
			void startTimer(stlTimer* sTimer);
			profile* stopTimer(stlTimer* sTimer);
		}
	}
}
#endif /* BENCH_H_ */
