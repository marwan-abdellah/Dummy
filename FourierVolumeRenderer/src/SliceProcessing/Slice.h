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


namespace Slice
{
    /* @ Creating th projection slice texture to be
     * attached to the FBO to have the RG slice projected into
     * it as a 2-component texture.
     */
    void createSliceTexture(int sliceWidth, int sliceHeight, GLuint* sliceTex_ID);

    void GetSlice(float sliceCenter, float sliceSideLength,
                  float rot_X, float rot_Y, float rot_Z,
                  GLuint* sliceTex_ID,
                  GLuint* spectralVolTex_ID,
                  GLuint sliceFBO_ID);

    void readBackSlice(int sliceWidth, int sliceHeight, GLuint sliceFBO_ID,
                              float* FBarray, fftwf_complex* complexSlice );

    void backTransformSlice(unsigned char* RecImage, float** Img_2D_Temp, float** Img_2D,
    fftwf_complex* complexSlice, float* AbsoluteReconstructedImage);

    void UploadImage(int sliceWidth, int sliceHeight, unsigned char* image, GLuint* sliceTex_ID);



}

#endif // SLICE_H
