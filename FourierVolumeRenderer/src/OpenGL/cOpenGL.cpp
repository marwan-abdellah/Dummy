#include "cOpenGL.h"
#include "OpenGLCheck.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

using namespace Magick;

/*************/
/* @ EXTERNS */
/*************/
extern float eXRot_Glob;
extern float eYRot_Glob;
extern float eZRot_Glob;
extern float eZoomLevel_Glob ;
extern float eSliceTrans_Glob ;
extern float eNormValue_Glob;
extern Image* GetSpectrumSlice();

/************/
/* @ LOCALS */
/************/
GLuint* cGL_ImageTexture_ID;

int    eWinWidth;
int    eWinHeight;
float  eImageZoom       = 1;
float  eNormValue       = 1.0;
int    eGloWinWidth     = 512;
int    eGloWinHeight    = 512;
Image* eDumpImage;

void OpenGL::updateSliceTexture(GLuint* iImageTexture_ID)
{
    cGL_ImageTexture_ID = iImageTexture_ID;
}

void OpenGL::prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture_ID)
{
    INFO("Preparing FBO");

    glGenFramebuffersEXT(1, iFBO_ID);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *iFBO_ID);

    /* @ Attaching the FBO to the associated texture */
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, *iSliceTexture_ID, 0);

    /* @ Unbinding */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    INFO("Preparing FBO DONE");
}

void OpenGL::initOpenGLContext(int argc, char** argv)
{
    INFO ("Initializing OpenGL Contex");

    /* checking the avialability of OpenGL context */
    if (isOpenGLAvailable())
    {
        INFO("OpenGL device is available");
    }
    else
    {
        INFO("OpenGL device is NOT available");
        EXIT(0);
    }

    /* @ GLUT Initialization */
    initGlut(argc, argv);

    /* Initialize necessary OpenGL extensions */
    if (!checkGLExtensions())
    {
        INFO("Missing OpenGL Necessary Extensions");
        EXIT(0);
    }
    else
        INFO("Requied OpenGL extensions are FOUND");

    INFO("Registering OpenGL callbacks");

    /* @ Registering OpenGL CallBack Functions */
    registerOpenGLCallBacks();

    INFO("Initializing OpenGL Contex DONE");
}

void OpenGL::initGlut(int argc, char** argv)
{
    /* @ Initializing GLUT */
    INFO("Initializing GLUT");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(eGloWinWidth, eGloWinHeight);
    glutCreateWindow("Fourier Volume Renderer - Marwan Abdellah");

    INFO("Display Mode : GLUT_RGBA | GLUT_DOUBLE");
    INFO("Initializing GLUT DONE");
}

bool OpenGL::checkGLExtensions()
{
    INFO("Checking OpenGL Extensions - GLEW");

    /* @ initializing GLEW */
    glewInit();

    /* Check OpenGL 2.0*/
    if (! glewIsSupported("GL_VERSION_2_0"))
    {
        INFO("ERROR: Support for necessary OpenGL extensions missing ");
        return 0;
    }
    else
        return 1;
}

void OpenGL::initOpenGL()
{
    /* @ Clearing color buffer */
    glClearColor (0.0, 0.0, 0.0, 0.0);

    /* Setting the pixel storage mode */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void OpenGL::displayGL()
{
    /* @ Clearing color buffer */
    glClear(GL_COLOR_BUFFER_BIT);

    /* @ Disabling depth test */
    glDisable(GL_DEPTH_TEST);

    /* @ Binding slice texture to be displayed On OpenGL Quad */
    glBindTexture(GL_TEXTURE_2D, *cGL_ImageTexture_ID);
    glEnable(GL_TEXTURE_2D);

    /* Adjusting slice texture parameters */
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* @ Adjusting viewport */
    glViewport(-eWinWidth / 2, -eWinHeight / 2, eWinWidth * 2, eWinHeight * 2);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    /* @ Center slice texture at the orgin (0,0) */
    glScalef(eImageZoom, eImageZoom, 1);
    glTranslatef(-0.5, -0.5, 0.0);

    /* @ Texture the slice on the QUAD */
    glBegin(GL_QUADS);
        glVertex2f(0, 0);		glTexCoord2f(0, 0);
        glVertex2f(0, 1);		glTexCoord2f(1, 0);
        glVertex2f(1, 1);		glTexCoord2f(1, 1);
        glVertex2f(1, 0);		glTexCoord2f(0, 1);
    glEnd();
    glPopMatrix();

    /* @ Release texture reference & disable texturing */
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    /* @ Swapping buffer contents */
    glutSwapBuffers();
}

void OpenGL::reshapeGL(int iWinWidth, int iWinHeight)
{
    /* @ Adjusting viewPort */
    glViewport(0, 0, iWinWidth, iWinHeight);

    /* @ For adjusting window size */
    eWinHeight = iWinHeight;
    eWinWidth = iWinWidth;

    // Projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void OpenGL::idleGL()
{
    glutPostRedisplay();
}

void OpenGL::keyboardGL(unsigned char fKey, int fX, int fY)
{

    // Dummy
    if (fX | fY | fKey) {}

    switch(fKey)
    {
        /* @ exit key */
        case 27:
            INFO("EXETING");
            EXIT(0);
            break;

        /* @ Rotation */
        case 'Q':
            eXRot_Glob += 1.0;
            INFO("X-axis rotaion angle : " + FTS(eXRot_Glob));
            break;
        case 'q':
            eXRot_Glob -= 1.0;
            INFO("X-axis rotaion angle : " + FTS(eXRot_Glob));
            break;
        case 'W':
            eYRot_Glob += 1.0;
            INFO("Y-axis rotaion angle : " + FTS(eYRot_Glob));
            break;
        case 'w':
            eYRot_Glob -= 1.0;
            INFO("Y-axis rotaion angle : " + FTS(eYRot_Glob));
            break;
        case 'E':
            eZRot_Glob += 1.0;
            INFO("Z-axis rotaion angle : " + FTS(eZRot_Glob));
            break;
        case 'e':
            eZRot_Glob -= 1.0;
            INFO("Z-axis rotaion angle : " + FTS(eZRot_Glob));
            break;

         /* @ Scaling & normalization */
        case 'R':
            eNormValue_Glob = eNormValue_Glob * 10;
            INFO("Scaling value : " + FTS(eNormValue_Glob));
            break;

        case 'r':
            eNormValue_Glob = eNormValue_Glob / 10;
            INFO("Scaling value : " + FTS(eNormValue_Glob));
            break;

        case 'T':
            eNormValue_Glob = eNormValue_Glob * 0.5;
            INFO("Scaling value : " + FTS(eNormValue_Glob));
            break;

        case 't':
            eNormValue_Glob = eNormValue_Glob / 0.5;
            INFO("Scaling value : " + FTS(eNormValue_Glob));
            break;

        case 'F':
            eSliceTrans_Glob = eSliceTrans_Glob + 0.00390625;
            INFO("Slice position : " + FTS(eSliceTrans_Glob));
            break;

        case 'f':
            eSliceTrans_Glob = eSliceTrans_Glob - 0.00390625;
            INFO("Slice position : " + FTS(eSliceTrans_Glob));
            break;

        case 's':
            eZoomLevel_Glob *= 5;
            printf("eZoomLevel_Glob : %f \n", eZoomLevel_Glob);
            break;

        case 'S':
            eZoomLevel_Glob /= 5;
            printf("eZoomLevel_Glob : %f \n", eZoomLevel_Glob);
            break;

        case 'a':
            eZoomLevel_Glob += 10;
            printf("eZoomLevel_Glob : %f \n", eZoomLevel_Glob);
            break;

        case 'A':
            eZoomLevel_Glob -= 10;
            printf("eZoomLevel_Glob : %f \n", eZoomLevel_Glob);
            break;

        case 'z' : eImageZoom += 0.5;
        break;

        case 'Z' : eImageZoom -= 0.5;
        break;

        default:
            break;
    }

    /* @ Reslice & redisplay */
    eDumpImage = GetSpectrumSlice();
    glutPostRedisplay();
}

void OpenGL::mouseGL(int fButton, int fState, int fX, int fY)
{
    if(fState == GLUT_DOWN)
    {
        if(fButton == GLUT_LEFT_BUTTON)
        {
            printf("1");
        }
        else if(fButton == GLUT_MIDDLE_BUTTON)
        {
             printf("12");
        }
        else if(fButton == GLUT_RIGHT_BUTTON)
        {
             printf("13");
        }
    }
    else
    {
        //alternate code
    }

    /* @ Reslice & redisplay */
    glutPostRedisplay();
}

void OpenGL::mouseMotionGL(int iX, int iY)
{
    // Dummy
    if (iX | iY) {}

    glutPostRedisplay();
}

void OpenGL::registerOpenGLCallBacks()
{
    /* Registering OpenGL context callbacks*/
    INFO("Registerng OpenGL context callbacks");

    glutDisplayFunc(displayGL);
    glutKeyboardFunc(keyboardGL);
    glutReshapeFunc(reshapeGL);
    glutIdleFunc(idleGL);
    glutMouseFunc(mouseGL);
    glutMotionFunc(mouseMotionGL);

    INFO("Registerng OpenGL context callbacks DONE");
}
