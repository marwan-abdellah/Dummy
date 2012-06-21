/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Typedefs.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "Logging.h"
#include "LoggingMACROS.h"


namespace Utils
{
    // Log file stream
    ostream_t logFileStream;
}
/**********************************************************
 *
 **********************************************************/
int Utils::createLogFile(string_t fileName)
{
    // Appending the extension string to the file name
    string_t logFileExtension = ".log";
    string_t logFileString = fileName + logFileExtension;

    // Open the log file stream
    Utils::logFileStream.open((char*)logFileString.c_str());

    // Check for a valid output file stream
    if (Utils::logFileStream != NULL)
    {
        // Logging
        LOG();
        return 0;
    }
    else
        return -1;
}

/**********************************************************
 *
 **********************************************************/

int Utils::closeLogFile()
{
    // Logging
    LOG();

    // Close the file stream
    Utils::logFileStream.close();

    // Check proper closure of the file
    if (Utils::logFileStream == 0)
        return 0;
    else
        return -1;
}


