/*
 * Bench.cpp
 *
 *  Created on: Jun 17, 2012
 *      Author: abdellah
 */

#include "Bench.h"

void Utils::Bench::STL::startTimer(stlTimer* sTimer)
{

}

profile* Utils::Bench::STL::stopTimer(stlTimer* sTimer)
{


	profile* sProfile = MEM_ALLOC_1D_GENERIC(profile, 1);

	return sProfile;
}
