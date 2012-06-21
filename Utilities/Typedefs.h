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

#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

typedef unsigned char uchar;
typedef unsigned char uimage;
typedef char image;
typedef std::string string_t ;
typedef std::ifstream istream_t;
typedef std::ofstream ostream_t;
typedef std::stringstream sstream_t;
typedef istream_t* istream_p;
typedef ostream_t* ostream_p;

#define COUT std::cout
#define ENDL std::endl
#define STRG std::string
#define TAB  "       "
#define LINE "_________________________________________________" << ENDL

#endif /* _TYPEDEFS_H_ */
