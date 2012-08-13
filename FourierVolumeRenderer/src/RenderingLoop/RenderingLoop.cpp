#include "RenderingLoop.h"
#include "OpenGL/cOpenGL.h"
#include "SliceProcessing/Slice.h"

/*****************************/
/* @ LOCALS - (e) = EXTERNAL */
/*****************************/
float*          eFB;
fftwf_complex*  eSlice_complex;
unsigned char*  eRecImage;
float*          eRecImageAbsolute;
float**         eImage_MAIN;
float**         eImage_TEMP;

void RenderingLoop::prepareRenderingArray(const int iSliceWidth,
                                          const int iSliceHeight)
{
    /* @ Reading back the FBO after initialization */
    eFB = (float*) malloc (iSliceWidth * iSliceHeight * sizeof(float));
    eSlice_complex = (fftwf_complex*) fftwf_malloc
            (iSliceWidth * iSliceHeight * sizeof(fftwf_complex));

    eRecImage = (unsigned char*) malloc (iSliceWidth * iSliceHeight * sizeof(char));
    eRecImageAbsolute = (float*) malloc (iSliceWidth * iSliceHeight * sizeof(float));

    /* 2D arrays for the wrapping around operations */
    eImage_MAIN = (float**) malloc (iSliceWidth * sizeof(float*));
    eImage_TEMP = (float**) malloc (iSliceWidth * sizeof(float*));
    for (int i = 0; i < iSliceWidth; i++)
    {
        eImage_MAIN[i] = (float*) malloc(iSliceHeight * sizeof(float));
        eImage_TEMP[i] = (float*) malloc(iSliceHeight * sizeof(float));
    }
}

void RenderingLoop::run(const float iRot_X,
                        const float iRot_Y,
                        const float iRot_Z,
                        float iSliceCenter, const float iSliceSideLength,
                        const int iSliceWidth, const int iSliceHeight,
                        GLuint* iSliceTexture_ID,
                        GLuint* iVolumeTexture_ID,
                        GLuint iFBO_ID,
                        GLuint* iImageTexture_ID)
{
    /* @ Extrat the projection slice from the spectral volume texture */
    Slice::getSlice(iSliceCenter, iSliceSideLength, iRot_X, iRot_Y, iRot_Z,
                    iSliceTexture_ID, iVolumeTexture_ID, iFBO_ID);

    /* @ Initialzing the eFB array */
    for (int i = 0; i < iSliceWidth * iSliceHeight * 2; i++)
        eFB[i] = 0;

    /* Allocating complex slice array */
    eSlice_complex = (fftwf_complex*)
            fftwf_malloc (iSliceWidth * iSliceHeight * sizeof(fftwf_complex));

    /* @ Reading back the extracted slice from the texture
     * attached to the FBo to the eSlice_complex array */
    Slice::readBackSlice(iSliceWidth, iSliceHeight, iFBO_ID, eFB, eSlice_complex);

    /* @ Back transform the extracted slice to create the projection */
    Slice::backTransformSlice(eRecImage, eImage_TEMP, eImage_MAIN, iSliceWidth, iSliceHeight,
    eSlice_complex, eRecImageAbsolute);

    /* @ Update the rendering context with the new image */
    cOpenGL::updateSliceTexture(iImageTexture_ID);

    /* @ Upload the image to the GPU */
    Slice::uploadImage(iSliceWidth, iSliceHeight, eRecImage, iImageTexture_ID);
}
