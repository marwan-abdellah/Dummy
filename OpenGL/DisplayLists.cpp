/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: OpenGL.cpp
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "DisplayLists.h"

void OpenGL::setDisplayList()
{
	// Display list ID
	GLuint displayList;

	// Central slice
	float centerPoint		= 0;

	// Left & right sides
	float sideLenght		= 0.5;

	// Central slice
	int numberSlices		= 1;

	// Number of vertexes
	int numberElements		= 4 * numberSlices;

	// Coordinates
	GLfloat *vPoints	= new GLfloat [3 * numberElements];
	GLfloat *ptr		= vPoints;

	// Fill the display list
	*(ptr++) = -sideLenght;
	*(ptr++) = -sideLenght;
	*(ptr++) =  centerPoint;

	*(ptr++) =  sideLenght;
	*(ptr++) = -sideLenght;
	*(ptr++) =  centerPoint;

	*(ptr++) =  sideLenght;
	*(ptr++) =  sideLenght;
	*(ptr++) =  centerPoint;

	*(ptr++) = -sideLenght;
	*(ptr++) =  sideLenght;
	*(ptr++) =  centerPoint;

	// Fill the display list (VERTEX_ARRAY)
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vPoints);

	// Create the display list, fill it and upload it
	displayList = glGenLists(1);
	glNewList(displayList, GL_COMPILE);
	glDrawArrays(GL_QUADS, 0, numberElements);
	glEndList();

	// Delete the CPU copy of the display list
	delete [] vPoints;
}
