#ifndef DISPLAYLIST_H
#define DISPLAYLIST_H

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "shared.h"

/* @ OpenGL namespace */
namespace OpenGL
{
    /* @ */
    GLuint setDisplayList(float center, float sideLength);
}

#endif // DISPLAYLIST_H
