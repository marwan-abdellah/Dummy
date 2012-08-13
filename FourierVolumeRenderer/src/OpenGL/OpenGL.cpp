#include "OpenGL.h"
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "shared.h"


extern void GetSpectrumSlice();


void OpenGL::prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture_ID)
{
    printf("Preparing FrameBuffer Object & Its Associated Texture \n");

    glGenFramebuffersEXT(1, iFBO_ID);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *iFBO_ID);

    // Attach Texture to FBO Color Attachement Point
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, *iSliceTexture_ID, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    printf("	Frame Buffer Preparation Done Successfully \n\n");
}

void OpenGL::InitOpenGLContext(int argc, char** argv)
{
    printf ("Initializing OpenGL Contex ... \n");
    printf ("	First Initialize OpenGL Context, So We Can Properly Set the GL for CUDA. \n");
    printf ("	This is Necessary in order to Achieve Optimal Performance with OpenGL/CUDA Interop. \n");


    // GLUT Initialization
    initGlut(argc, argv);



    // Register OpenGL CallBack Functions
    OpenGLRegisterOpenGLCallBacks();

}

extern void GetSpectrumSlice();
extern  GLuint		miSliceTexture_IDID; 	// Extracted Slice ID
GLuint* imageTexure_ID;

namespace OpenGL
{
    int dispWinWidth = 512;
    int dispWinHeight = 512;
    int rot_X = 0;
    int rot_Y = 0;
    int rot_Z = 0;
    int scaleFactor = 1;

    extern float mXrot = 0;
    extern float mYrot = 0;
    extern float mZrot = 0;
    extern int mScalingFactor = 1;
    int sVal;

    int sWindowWidth;
    int sWindowHeight;

    float mImageScale;
}


void OpenGL::updateImageTexture(GLuint* newImageTexture_ID)
{
    imageTexure_ID = newImageTexture_ID;
}

void OpenGL::initGlut(int argc, char** argv)
{
    // Initializing GLUT
    printf("Initializing GLUT ... \n");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(dispWinWidth, dispWinHeight);
    glutCreateWindow("Fourier Volume Rendering on CUDA");

    printf("	Display Mode		: GLUT_RGBA | GLUT_DOUBLE \n");
    printf("	GLUT Windows Size	: %d %d \n \n", dispWinWidth, dispWinHeight);
}

void OpenGL::initOpenGL()
{
    printf("Initializing OpenGL ... \n");

    // Clearing Buffer
    glClearColor (0.0, 0.0, 0.0, 0.0);

    // Pixel Storage Mode
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    printf("	Initializing OpenGL Done \n\n");
}

void OpenGL::DisplayGL()
{
    // Clearing color buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Disable depth buffer
    glDisable(GL_DEPTH_TEST);

    // Binding slice texture to be displayed on OpenGL polygon
    glBindTexture(GL_TEXTURE_2D, *imageTexure_ID);
    glEnable(GL_TEXTURE_2D);

    // Slice texture parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Adjusting Viewport
    glViewport(-sWindowWidth / 2, -sWindowHeight / 2, sWindowWidth * 2, sWindowHeight * 2);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Center Slice Texture (0,0)
    glScalef(mImageScale, mImageScale, 1);
    glTranslatef(-0.5, -0.5, 0.0);

    glBegin(GL_QUADS);
        glVertex2f(0, 0);		glTexCoord2f(0, 0);
        glVertex2f(0, 1);		glTexCoord2f(1, 0);
        glVertex2f(1, 1);		glTexCoord2f(1, 1);
        glVertex2f(1, 0);		glTexCoord2f(0, 1);
    glEnd();
    glPopMatrix();

    // Release Texture Reference & Disable Texturing
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // Swapping Buffer Contents
    glutSwapBuffers();
}


void OpenGL::Reshape(int windowWidth, int windowHight)
{
    // Adjust your viewport
    glViewport(0, 0, windowWidth, windowHight);

    // Update global window parameters to reflect texture updates
    sWindowHeight = windowHight;
    sWindowWidth = windowWidth;

    // Adjust the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void OpenGL::Idle()
{
    // Redisplay loop
    glutPostRedisplay();
}

void OpenGL::KeyBoard(unsigned char fKey, int fX, int fY)
{

    // Dummy
    if (fX | fY | fKey) {}

    switch(fKey)
    {
        case 27:
            exit (0);
            break;
        case 'Q':
            mXrot += 5.0;
            printf("Rotating %f around mLoop ... \n", (float) mXrot);
            break;
        case 'q':
            mXrot -= 5.0;
            printf("Rotating %f around mLoop ... \n", (float) mXrot);
            break;
        case 'W':
            mYrot += 5.0;
            printf("Rotating %f around Y ... \n", (float) mYrot);
            break;
        case 'w':
            mYrot -= 5.0;
            printf("Rotating %f around Y ... \n", (float) mYrot);
            break;
        case 'E':
           mZrot += 5.0;
            printf("Rotating %f around Z ... \n", (float) mZrot);
            break;
        case 'e':
            mZrot -= 5.0;
            printf("Rotating %f around Z ... \n", (float) mZrot);
            break;
        case ' ':
           // CUDA_ENABLED = (!CUDA_ENABLED);
            printf("Enabling / Disabling CUDA Processing");
            break;

        case 'R':
            sVal = sVal * 10;
            printf("sVal %f \n", sVal);
            break;

        case 'r':
            sVal = sVal / 10;
            printf("sVal %d \n", sVal);
            break;

        case 'o':
            //trans = trans + 1;
            //printf("trans : %f/256 \n", trans);
            break;

        case 'p':
            //trans = trans - 1;
            //printf("trans : %f/256 \n", trans);
            break;

        case 's':
            mScalingFactor *= 5;
            printf("mScalingFactor : %d \n", mScalingFactor);
            break;

        case 'S':
            mScalingFactor /= 5;
            printf("mScalingFactor : %d \n", mScalingFactor);
            break;

        case 'a':
            mScalingFactor += 10;
            printf("mScalingFactor : %d \n", mScalingFactor);
            break;

        case 'A':
            mScalingFactor -= 10;
            printf("mScalingFactor : %d \n", mScalingFactor);
            break;

        case 'z' :mImageScale += 0.5;
        break;

        case 'Z' :mImageScale -= 0.5;
        break;

        default:
            break;
    }

    GetSpectrumSlice();
    glutPostRedisplay();
}


void OpenGL::Mouse(int fButton, int fState, int fX, int fY)
{
    glutPostRedisplay();

    // Dummy
    if (fX | fY | fState | fButton) {}

}

void OpenGL::MouseMotion(int fX, int fY)
{
    glutPostRedisplay();

    // Dummy
    if (fX | fY) {}
}

void OpenGL::OpenGLRegisterOpenGLCallBacks()
{
    // Registering OpenGL Context
    printf("Registerng OpenGL Context CallBacks ... \n");

    glutDisplayFunc(OpenGL::DisplayGL);
    glutKeyboardFunc(OpenGL::KeyBoard);
    glutReshapeFunc(OpenGL::Reshape);
    glutIdleFunc(OpenGL::Idle);
    glutMouseFunc(OpenGL::Mouse);
    glutMotionFunc(OpenGL::MouseMotion);

    printf("	CallBacks Registered Successfully \n\n");
}
