#ifndef OPENGL_H
#define OPENGL_H

#include <shared.h>
// OpenGL Includes
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>


namespace OpenGL
{
    void InitOpenGLContext(int argc, char** argv);
    void initGlut(int argc, char** argv);
    void initOpenGL();
    void DisplayGL();
    void Reshape(int windowWidth, int windowHight);
    void Idle();
    void KeyBoard(unsigned char fKey, int fX, int fY);
    void Mouse(int fButton, int fState, int fX, int fY);
    void MouseMotion(int fX, int fY);
    void OpenGLRegisterOpenGLCallBacks();
    void updateImageTexture(GLuint* newImageTexture_ID);
    void prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture);


}


#endif // OPENGL_H
