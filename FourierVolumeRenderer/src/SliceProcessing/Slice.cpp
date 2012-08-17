#include "Slice.h"
#include "OpenGL/DisplayList.h"
#include "WrappingAround/WrappingAround.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

extern float eNormValue_Glob;
extern float eSliceUniSize_Glob;

void Slice::getSlice(const float iSliceCenter,
                     const float iSliceSideLength,
                     const float rot_X,
                     const float rot_Y,
                     const float rot_Z,
                     GLuint* iSliceTexture_ID,
                     GLuint* spectralVolTex_ID,
                     GLuint iFOB_ID)
{
    /* @ Creating the display list for the proxy geometry */
    GLuint eDisplayList = OpenGL::setDisplayList(iSliceCenter, iSliceSideLength);

    /* @ Render to FBO render target */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, iFOB_ID);

    /* Clear color buffer */
    glClear(GL_COLOR_BUFFER_BIT);

    /* @ Enable 3D texturing */
    glEnable(GL_TEXTURE_3D);

    /* @ setting up texture variables */
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    /* @ Binding 3D texture */
    glBindTexture(GL_TEXTURE_3D, *spectralVolTex_ID);

    /* @ Adjusting OpenGL Viewport to fit the slice size */
    glViewport(-(eSliceUniSize_Glob / 2),-(eSliceUniSize_Glob / 2),
               (eSliceUniSize_Glob * 2),(eSliceUniSize_Glob * 2));

    /* Texture corrdinate automatic generation */
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);
    glEnable(GL_TEXTURE_GEN_R);

    /* @ Adjusting the PROJECTION & MODELVIEW matricies */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /* Defining the 6 main clip planes */
    static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
    static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
    static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
    static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
    static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
    static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};

    /* @ Define equations for automatic texture coordinate generation */
    static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
    static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
    static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

    /* @ Save state */
    glPushMatrix ();

    /* @ Transform (Rotation Only) the Viewing Direction
     * We don't need except the - 0.5 translation in each dimension to adjust
     * the texture in the center of the scene
     */
    glRotatef(-rot_X, 0.0, 0.0, 1.0);
    glRotatef(-rot_Y, 0.0, 1.0, 0.0);
    glRotatef(-rot_Z , 1.0, 0.0, 0.0);
    glTranslatef(-0.5, -0.5, -0.5);

    /* @ Automatic texture coord generation */
    glTexGenfv(GL_S, GL_EYE_PLANE, x);
    glTexGenfv(GL_T, GL_EYE_PLANE, y);
    glTexGenfv(GL_R, GL_EYE_PLANE, z);

    /* @ Define the 6 basic clipping planes (of the UNITY CUBE) */
    glClipPlane(GL_CLIP_PLANE0, eqx0);
    glClipPlane(GL_CLIP_PLANE1, eqx1);
    glClipPlane(GL_CLIP_PLANE2, eqy0);
    glClipPlane(GL_CLIP_PLANE3, eqy1);
    glClipPlane(GL_CLIP_PLANE4, eqz0);
    glClipPlane(GL_CLIP_PLANE5, eqz1);

    glPopMatrix ();

    /* @ Enabling clip planes */
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);

    glEnable(GL_CLIP_PLANE2);
    glEnable(GL_CLIP_PLANE3);

    glEnable(GL_CLIP_PLANE4);
    glEnable(GL_CLIP_PLANE5);

    /* @ Rendering enclosing rectangle (only at (0,0) plane) */
    glCallList(eDisplayList);
    glPopMatrix();

    /* @ Disable 3D texturing & clipping planes */
    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_GEN_R);
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);

    /* @ Unbind the FBO */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    /* @ Render using the texture target */
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    /* Enable 2D texturing */
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

    /* @ project the extracted slice to the slice texture target */
    glPushMatrix();
    glLoadIdentity();
    glBegin(GL_QUADS);
        glNormal3f(0.0f, 0.0f, 1.0);
        glTexCoord2f(0.0,0.0);		glVertex3f(0.0,0.0,0.0);
        glTexCoord2f(1.0,0.0);		glVertex3f(1.0,0.0,0.0);
        glTexCoord2f(1.0,1.0);		glVertex3f(1.0,1.0,0.0);
        glTexCoord2f(0.0,1.0);		glVertex3f(0.0,1.0,0.0);
    glEnd();
    glPopMatrix();

    // Disable 2D texturig
    glDisable(GL_TEXTURE_2D);

    // glDeleteLists(eDisplayList, 0);
}

void Slice::readBackSlice(const int iSliceWidth, const int iSliceHeight,
                          GLuint iFOB_ID,
                          float* iSlice_FB, fftwf_complex* iSlice_complex)
{
    /* @ Binding FBO to read from it */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, iFOB_ID);
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

    /* @ Writing FBO texture components to the iSlice_FB array */
    glReadPixels(0, 0, iSliceWidth, iSliceHeight, RG, GL_FLOAT, iSlice_FB);

    /* @ Unbinding the FBO */
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    /* @ Saving the extracted slice in a complex array for later handling */
    int ctr = 0;
    for (int i = 0; i < (iSliceWidth * iSliceHeight * 2); i += 2)
    {
        iSlice_complex[ctr][0] = iSlice_FB[i];
        iSlice_complex[ctr][1] = iSlice_FB[i + 1];
        ctr++;
    }
}

void Slice::backTransformSlice(unsigned char *iRecImage,
                               float** iSquareImage_TEMP,
                               float** iSquareImage_MAIN,
                               const int iSliceWidth,
                               const int iSliceHeight,
                               fftwf_complex* iSlice_complex,
                               float* iRecImage_ABS)
{
    /* @ 2D iFFT for the projection slice to get the final projection */
    fftwf_plan eFFTPlan;
    eFFTPlan = fftwf_plan_dft_2d(iSliceWidth, iSliceHeight, iSlice_complex,iSlice_complex , FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(eFFTPlan);

    /* Scaling the reconstructed iRecImage */
    int eSliceSize = iSliceWidth * iSliceHeight;
    int eNormValue = eSliceSize * 100 * eNormValue_Glob;
    for (int i = 0; i < eSliceSize; i++)
        iRecImage_ABS[i] = (float)
                sqrt((iSlice_complex[i][0] * iSlice_complex[i][0]) +
                     (iSlice_complex[i][1] * iSlice_complex[i][1])) / (eNormValue);


    /*@ Wrap around the final iRecImage to diaply it correctly */
    int ctr = 0;
    for (int i = 0; i < iSliceWidth; i++)
    {
        for(int j = 0; j < iSliceHeight; j++)
        {
            /* @ Just clearing the arrays */
            iSquareImage_TEMP[i][j] = iRecImage_ABS[ctr];
            iSquareImage_MAIN[i][j] = 0;
            ctr++;
        }
    }

    if (iSliceWidth == iSliceHeight)
    {
        INFO("Projection image has UNIFIED dimensions : " + ITS(iSliceWidth));
    }

    else
    {
        INFO("Projection image DOEN'T have UNIFIED dimensions - EXITING");
        EXIT(0);
    }
    WrappingAround::WrapAroundImage(iSquareImage_MAIN, iSquareImage_TEMP, iRecImage_ABS, iSliceWidth);

    /* @ Downscaling the pixel value to fit th BYTE range */
    for (int i = 0; i < iSliceWidth * iSliceHeight; i++)
        iRecImage[i] = (unsigned char)(iRecImage_ABS[i]);
}

void Slice::uploadImage(const int iSliceWidth, const int iSliceHeight,
                        const float* iRecImage, GLuint* iSliceTexture_ID)
{
    /* @ Create 2D texture object as a render target */
    glGenTextures(1, iSliceTexture_ID);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

    /*  Adusting the 2D exture parameters */
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* @ Automatic mipmap Generation included in OpenGL v1.4 */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                 iSliceWidth, iSliceHeight, 0,
                 GL_LUMINANCE, GL_FLOAT, iRecImage);

    /* @ Unbinding texture */
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Slice::createSliceTexture(int iSliceWidth, int iSliceHeight, GLuint* iSliceTexture_ID)
{
    INFO("Creating & binding slice texture");

    /* @ Create 2D texture object as a render target */
    glGenTextures(1, iSliceTexture_ID);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

    /* @ 2D Texture creation & parameters */
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* @ Uploading the slice texture to the GPU */
    glTexImage2D(GL_TEXTURE_2D, 0, RG32F, iSliceWidth, iSliceHeight, 0, RG, GL_FLOAT, NULL);

    /* @ Unbind textures */
    glBindTexture(GL_TEXTURE_2D, 0);

    INFO("Creating & binding slice texture DONE ");
}

