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

/* @ internal includes */
#include "shared.h"

namespace RenderingLoop
{
    /* @ */
    void prepareRenderingArray(const int iSliceWidth,
                               const int iSliceHeight);
    /* @ */
    Magick::Image* run(const float iRot_X,
                       const float iRot_Y,
                       const float iRot_Z,
                       float iSliceCenter,
                       const float iSliceSideLength,
                       const int iSliceWidth, const int iSliceHeight,
                       GLuint* iSliceTexture_ID,
                       GLuint* iVolumeTexture_ID,
                       GLuint iFBO_ID,
                       GLuint* iImageTexture_ID);
}

#endif // RENDERINGLOOP_H
