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

#ifndef _LOGGING_H_
#define _LOGGING_H_

#include "Typedefs.h"
#include "LoggingMACROS.h"


namespace Utils
{
	// Log file stream
	extern ostream_t logFileStream;

	int createLogFile(string_t fileName);
	void log();
	void logHeader(string_t Hdr);
	void logMsg(string_t Msg);
	int closeLogFile();
}


#endif /* _LOGGING_H_ */
