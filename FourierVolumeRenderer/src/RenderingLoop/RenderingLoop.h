#ifndef RENDERINGLOOP_H
#define RENDERINGLOOP_H

/* @ OpenGL includes */
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* @ System includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* @ FFTW includes */
#include <fftw3.h>

namespace RenderingLoop
{
    void prepareRenderingArray(int iSliceWidth, int iSliceHeight);
    void run(float iRot_X, float iRot_Y, float iRot_Z,
                            float iSliceCenter, float iSliceSideLength,
                            int iSliceWidth, int iSliceHeight,
                            GLuint* iSliceTexture_ID,
                            GLuint* iVolumeTexture_ID,
                            GLuint iFBO_ID,
                            GLuint* iImageTexture_ID);
}

#endif // RENDERINGLOOP_H

// RenderingLoop::run(mXrot, mYrot, mZrot, 0, 1, 256, 256,
  //                 &eSliceTexture_ID, &eVolumeTexture_ID, eFBO_ID, &eImageTexture_ID);
