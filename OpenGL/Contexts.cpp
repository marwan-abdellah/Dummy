/*
 * Contexts.cpp
 *
 *  Created on: May 23, 2012
 *      Author: abdellah
 */

#include "Contexts.h"
#include "OpenGL.h"

// void OpenGL::InitOpenGLContext(int argc, char** argv)
void OpenGL::initGLContext(int argc, char** argv)
{
	// printf ("Initializing OpenGL Contex ... \n");
	// printf ("	First Initialize OpenGL Context, So We Can Properly Set the GL for CUDA. \n");
	// printf ("	This is Necessary in order to Achieve Optimal Performance with OpenGL/CUDA Interop. \n");

	if (IsOpenGLAvailable(appName))
		fprintf(stderr, "	OpenGL Device is Available \n\n");
	else
	{
		fprintf(stderr, "	OpenGL device is NOT Available, [%s] exiting...\n  PASSED\n\n", appName );
		exit(0);
	}

	// GLUT Initialization
	initGlut(argc, argv);

	// Initialize Necessary OpenGL Extensions
	if (CUTFalse == CheckOpenGLExtensions())
	{
		printf("	Missing OpenGL Necessary Extensions  \n Exiting ..... \n");
		exit(0);
	}
	else
		printf("	Requied OpenGL Extensions are Found \n\n");

	// Register OpenGL CallBack Functions
	RegisterOpenGLCallBacks();
}

void OpenGL::initGlut(int argc, char** argv)
{
	// Initializing GLUT
	printf("Initializing GLUT ... \n");

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(mWindowWidth, mWindowHeight);
	glutCreateWindow("Fourier Volume Rendering on CUDA");

	printf("	Display Mode		: GLUT_RGBA | GLUT_DOUBLE \n");
	printf("	GLUT Windows Size	: %d mLoop %d \n \n", mWindowWidth, mWindowHeight);
}




CUTBoolean CheckOpenGLExtensions()
{
	printf("Checking OpenGL Extensions via GLEW ... \n");
	glewInit();

	if (! glewIsSupported("GL_VERSION_2_0"))
	{
		fprintf(stderr, "ERROR: Support for Necessary OpenGL Extensions Missing ..... \n");
		fflush(stderr);
		return CUTFalse;
    	}
    	else
    		return CUTTrue;
}

void initOpenGL()
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
	// Clearing Color Buffer
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	// Binding Slice Texture to be Displayed On OpenGL Quad
	glBindTexture(GL_TEXTURE_2D, mSliceTextureID);
	glEnable(GL_TEXTURE_2D);

	// Slice Texture Parameters
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

void OpenGL::reshapeWindow(int fWindowWidth, int fWindowHeight)
{
	// ViewPort
	glViewport(0, 0, fWindowWidth, fWindowHeight);
	sWindowHeight = fWindowHeight;
	sWindowWidth = fWindowWidth;

	// Projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
}

void OpenGL::idle()
{
	glutPostRedisplay();
}

void OpenGL::keyBorad(unsigned char fKey, int fX, int fY)
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
			CUDA_ENABLED = (!CUDA_ENABLED);
			printf("Enabling / Disabling CUDA Processing");
			break;

		case 'R':
			sVal = sVal * 10;
			printf("sVal %f \n", sVal);
			break;

		case 'r':
			sVal = sVal / 10;
			printf("sVal %f \n", sVal);
			break;

		case 'o':
			trans = trans + 1;
			printf("trans : %f/256 \n", trans);
			break;

		case 'p':
			trans = trans - 1;
			printf("trans : %f/256 \n", trans);
			break;

		case 's':
			mScalingFactor *= 5;
			printf("mScalingFactor : %f \n", mScalingFactor);
			break;

		case 'S':
			mScalingFactor /= 5;
			printf("mScalingFactor : %f \n", mScalingFactor);
			break;

		case 'a':
			mScalingFactor += 10;
			printf("mScalingFactor : %f \n", mScalingFactor);
			break;

		case 'A':
			mScalingFactor -= 10;
			printf("mScalingFactor : %f \n", mScalingFactor);
			break;

		case 'z' : mImageScale += 0.5;
		break;

		case 'Z' : mImageScale -= 0.5;
		break;

        default:
			break;
	}

	// Re-Slice & Re-Display
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



