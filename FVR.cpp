/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include <stdio.h>
#include <fftw3.h>
#include "Volume/Volume.h"
#include "Utilities/Typedefs.h"
#include "Utilities/MACROS.h"
#include "Utilities/Logging.h"
#include "Globals.h"
#include "Utilities/Memory.h"
#include "Utilities/Logging.h"
#include "Utilities/XL.h"
#include "Utilities/LoggingMACROS.h"
#include "FourierVolumeRenderer.h"
#include "Timers/TimerGlobals.h"
#include "Timers/BoostTimers.h"

#define SIZE 4

#include <iostream>

int main( int argc, char** argv )
{

    Utils::createLogFile("FVR_Log");
    LOG();

    //Volume::testVolume();

    Utils::Test_XL();

    FVR::doItAllHere();


    Utils::closeLogFile();
    return 0;
}
