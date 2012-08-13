#ifndef _COPENGL_H_
#define _COPENGL_H_


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <fftw3.h>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <cutil_inline.h>




using namespace std;

#define OFFSET(i) ((char *)NULL + (i))

// extern  GLuint		mSliceTextureID; 	// Extracted Slice ID

namespace cOpenGL
{
    void InitOpenGLContext(int argc, char** argv);
    void initOpenGL();
    CUTBoolean CheckOpenGLExtensions();
    void initGlut(int argc, char** argv);
    void DisplayGL();
    void Reshape(int fWindowWidth, int fWindowHeight);
    void KeyBoard(unsigned char, int, int);
    void Idle();
    void Mouse(int fButton, int fState, int fX, int fY);
    void MouseMotion(int fX, int fY);
    void RegisterOpenGLCallBacks();

    void prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture_ID);
    void updateSliceTexture(GLuint* iImageTexture_ID);
}

#endif // COPENGL_H
