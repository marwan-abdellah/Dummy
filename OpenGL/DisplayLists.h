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

#ifndef DISPLAY_LISTS_H_
#define DISPLAY_LISTS_H_

#include "OpenGL.h"
#include <GL/glew.h>
#include <GL/glu.h>

#include "DisplayLists.h"

#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
	#define USE_TEXSUBIMAGE2D
#else
	#include <GL/glut.h>
#endif

namespace OpenGL
{
	void setDisplayList();
}

#endif /* DISPLAY_LISTS_H_ */
