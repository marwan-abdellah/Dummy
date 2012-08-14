#ifndef SLICE_H
#define SLICE_H

/* @ OpenGL includes */
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* System includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* FFTW includes */
#include <fftw3.h>

/* Internal includes */
#include "shared.h"


/* @ Slice namespace */
namespace Slice
{
    /* @ Creating th projection slice texture to be
     * attached to the FBO to have the RG slice projected into
     * it as a 2-component texture.
     */
    void createSliceTexture(int sliceWidth, int sliceHeight, GLuint* sliceTex_ID);

    /* @ */
    void getSlice(const float iSliceCenter,
                         const float iSliceSideLength,
                         const float rot_X,
                         const float rot_Y,
                         const float rot_Z,
                         GLuint* iSliceTexture_ID,
                         GLuint* spectralVolTex_ID,
                         GLuint sliceFBO_ID);

    /* @ */
    void readBackSlice(const int iSliceWidth, const int iSliceHeight,
                       GLuint iFOB_ID,
                       float* iSlice_FB, fftwf_complex* iSlice_complex);
    /* @ */
    void backTransformSlice(unsigned char *iRecImage,
                            float** iSquareImage_TEMP,
                            float** iSquareImage_MAIN,
                            const int iSliceWidth,
                            const int iSliceHeight,
                            fftwf_complex* iSlice_complex,
                            float* iRecImage_ABS);

    /* @ */
    void uploadImage(const int iSliceWidth, const int iSliceHeight,
                    const float* iRecImage, GLuint* iSliceTexture_ID);
}

#endif // SLICE_H
