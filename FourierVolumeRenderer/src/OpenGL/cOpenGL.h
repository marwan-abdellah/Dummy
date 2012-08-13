#ifndef _COPENGL_H_
#define _COPENGL_H_

/* @ OpenGL includes */
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* System includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* @ CUDA includes */
#include <cutil_inline.h>

/* @ Internal includes */
#include "shared.h"

#define OFFSET(i) ((char *)NULL + (i))

/* @ OpenGL namespace */
namespace OpenGL
{
    /* @ initialization functions */
    void initOpenGLContext(int argc, char** argv);
    void initOpenGL();
    void initGlut(int argc, char** argv);

    /* @ Checking required OpenGL extensions */
    bool checkGLExtensions();

    /* @ preparing FBO */
    void prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture_ID);
    void updateSliceTexture(GLuint* iImageTexture_ID);

    /* @ OpenGL callbacks */
    void displayGL();
    void reshapeGL(int iWinWidth, int iWinHeight);
    void keyboardGL(unsigned char Button, int iX, int iY);
    void idleGL();
    void mouseGL(int iButton, int iState, int IX, int iY);
    void mouseMotionGL(int iX, int iY);
    void registerOpenGLCallBacks();
}

#endif // COPENGL_H
