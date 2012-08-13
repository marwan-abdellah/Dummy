#include "Slice.h"
#include "OpenGL/DisplayList.h"
#include "FFTShift/FFTShift.h"

void Slice::GetSlice(float sliceCenter,
                     float sliceSideLength,
                     float rot_X,
                     float rot_Y,
                     float rot_Z,
                     GLuint* iSliceTexture_ID,
                     GLuint* spectralVolTex_ID,
                     GLuint sliceFBO_ID)
{
    printf("Extracting Fourier Slice ... \n");

    GLuint displayList = OpenGL::setDisplayList(sliceCenter, sliceSideLength);

    // Render to Framebuffer Render Target
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, sliceFBO_ID);

    // Clear Buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable 3D Texturing
    glEnable(GL_TEXTURE_3D);

    // Setup Texture Variables & Parameters
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // Bind 3D Texture
    glBindTexture(GL_TEXTURE_3D, *spectralVolTex_ID);

    // Adjust OpenGL View Port
    glViewport(-128,-128,512,512);

    // Texture Corrdinate Generation
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);
    glEnable(GL_TEXTURE_GEN_R);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 6 Main - Clip Planes
    static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
    static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
    static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
    static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
    static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
    static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};

    // Define Equations for Automatic Texture Coordinate Generation
    static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
    static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
    static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

    glPushMatrix ();

    // Transform (Rotation Only) the Viewing Direction
    /*
     * We don't need except the - 0.5 translation in each dimension to adjust
     * the texture in the center of the scene
     */
    glRotatef(-rot_X, 0.0, 0.0, 1.0);
    glRotatef(-rot_Y, 0.0, 1.0, 0.0);
    glRotatef(-rot_Z, 1.0, 0.0, 0.0);
    glTranslatef(-0.5, -0.5, -0.5);

    // Automatic Texture Coord Generation.
    glTexGenfv(GL_S, GL_EYE_PLANE, x);
    glTexGenfv(GL_T, GL_EYE_PLANE, y);
    glTexGenfv(GL_R, GL_EYE_PLANE, z);

    // Define the 6 Basic Clipping Planes (of the UNITY CUBE)
    glClipPlane(GL_CLIP_PLANE0, eqx0);
    glClipPlane(GL_CLIP_PLANE1, eqx1);
    glClipPlane(GL_CLIP_PLANE2, eqy0);
    glClipPlane(GL_CLIP_PLANE3, eqy1);
    glClipPlane(GL_CLIP_PLANE4, eqz0);
    glClipPlane(GL_CLIP_PLANE5, eqz1);

    glPopMatrix ();

    // Enable Clip Planes
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);

    glEnable(GL_CLIP_PLANE2);
    glEnable(GL_CLIP_PLANE3);

    glEnable(GL_CLIP_PLANE4);
    glEnable(GL_CLIP_PLANE5);

    // Render Enclosing Rectangle (Only at (0,0) Plane)
    glCallList(displayList);
    glPopMatrix();

    // Disable Texturing
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

    // Unbind the Framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    // Render Using the Texture Target
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable 2D Texturing
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

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

    // Disable Texturig
    glDisable(GL_TEXTURE_2D);


}


void Slice::readBackSlice(int iSliceWidth, int iSliceHeight, GLuint sliceFBO_ID,
                          float* FBarray, fftwf_complex* complexSlice )
{
    // Attach FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, sliceFBO_ID);
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

    // Writing FBO Texture Components into mFrameBufferArray
    glReadPixels(0,0, iSliceWidth, iSliceHeight, RG, GL_FLOAT, FBarray);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    printf("CPU Processing \n");

    int ctr = 0;
    for (int i = 0; i < (256 * 256 * 2); i += 2)
    {
        complexSlice[ctr][0] = FBarray[i];
        complexSlice[ctr][1] = FBarray[i+1];
        ctr++;
    }
}


void Slice::backTransformSlice(unsigned char *RecImage, float** Img_2D_Temp, float** Img_2D, fftwf_complex* complexSlice, float* AbsoluteReconstructedImage)
{
    fftwf_plan plan;
    // 2D FFT
    // printf("Executing 2D Inverse FFT - Begin.... \n");
    plan = fftwf_plan_dft_2d(256, 256, complexSlice,complexSlice , FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    //* printf("Executing 2D Inverse FFT - End.... \n");

    // Scaling
    int mSliceSize = 256 * 256;
    int mNormalizedVal = mSliceSize * 50 * 1;
    for (int sliceCtr = 0; sliceCtr < mSliceSize; sliceCtr++)
    {
        AbsoluteReconstructedImage[sliceCtr] =
                (float) sqrt((complexSlice[sliceCtr][0] * complexSlice[sliceCtr][0]) + (complexSlice[sliceCtr][1] * complexSlice[sliceCtr][1]))/(mNormalizedVal);
    }

    int ctr = 0;
    for (int i = 0; i < 256; i++)
        for(int j = 0; j < 256; j++)
        {
            Img_2D_Temp[i][j] = AbsoluteReconstructedImage[ctr];
            Img_2D[i][j] = 0;
            ctr++;
        }

    //* printf("Wrapping Around Resulting Image - Begin.... \n");
    Img_2D = FFTShift::FFT_Shift_2D(Img_2D_Temp, Img_2D, 256);
    AbsoluteReconstructedImage = FFTShift::Repack_2D(Img_2D, AbsoluteReconstructedImage, 256);
    //* printf("Wrapping Around Resulting Image - End.... \n");

    for (int i = 0; i < 256 * 256; i++)
    {
        RecImage[i] = (char)(AbsoluteReconstructedImage[i]);
    }
}


void Slice::UploadImage(int iSliceWidth, int iSliceHeight, unsigned char* image, GLuint* iSliceTexture_ID)
{
    // Create 2D Texture Object as a Render Target
    glGenTextures(1, iSliceTexture_ID);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

    // 2D Texture Creation & Parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Automatic Mipmap Generation Included in OpenGL v1.4
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, iSliceWidth, iSliceHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image);

    glBindTexture(GL_TEXTURE_2D, 0);
}


void Slice::createSliceTexture(int iSliceWidth, int iSliceHeight, GLuint* iSliceTexture_ID)
{
    printf("Creating & Binding Input Slice Texture ... \n");

    // Create 2D Texture Object as a Render Target
    glGenTextures(1, iSliceTexture_ID);
    glBindTexture(GL_TEXTURE_2D, *iSliceTexture_ID);

    // 2D Texture Creation & Parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Automatic Mipmap Generation Included in OpenGL v1.4
    glTexImage2D(GL_TEXTURE_2D, 0, RG32F, iSliceWidth, iSliceHeight, 0, RG, GL_FLOAT, NULL);

    // Unbind Texture
    glBindTexture(GL_TEXTURE_2D, 0);

    printf("	Creating Input Slice Texture Done Successfully \n\n");
}

