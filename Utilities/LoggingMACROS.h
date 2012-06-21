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

#ifndef _LOGGING_MACROS_H_
#define _LOGGING_MACROS_H_

#include "Typedefs.h"
#include "Logging.h"

#define fileStream Utils::logFileStream

#define LOG() 												\
	fileStream << "- FILE:" << STRG(__FILE__) << 			\
	":[" << (__LINE__) << "]:" << ENDL << 					\
	"--- FUNCTION: " << STRG(__FUNCTION__) << ENDL << 		\
	"@ " << "[" << STRG(__DATE__) << ", " << 				\
	STRG(__TIME__) << "]" << ENDL << LINE;

#define LOG_HDR(Header) 									\
	fileStream << "- FILE:" << STRG(__FILE__) << 			\
	":[" << (__LINE__) << "]:" << ENDL << 					\
	"--- FUNCTION: " << STRG(__FUNCTION__) << ENDL <<		\
	"----- HEADER:  " << Header << ENDL << LINE ;

#define LOG_MSG(Message) 									\
	fileStream << "- FILE:" << STRG(__FILE__) << 			\
	":[" << (__LINE__) << "]:" << ENDL << 					\
	"--- FUNCTION: " << STRG(__FUNCTION__) << ENDL <<		\
	"----- MESSAGE: " << Message << ENDL << LINE;


#endif /* _LOGGING_MACROS_H_ */
