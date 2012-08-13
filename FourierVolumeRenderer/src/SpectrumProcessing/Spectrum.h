#ifndef SPECTRUM_H
#define SPECTRUM_H

/* @ Internal includes */
#include "shared.h"

/* @ System includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* @ FFTW includes */
#include <fftw3.h>

/* @ OpenGL includes */
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

/*@ spectrum namespace */
namespace Spectrum
{
    /* @ */
    fftwf_complex* createSpectrum(volume* iSpectralVolume);

    /* @ */
    float* packingSpectrumTexture(const fftwf_complex* iSpectralVolume,
                                  const volDim* iVolDim);

    /* @ */
    void uploadSpectrumTexture(GLuint* iSpectralTexture_ID,
                               const float* iSpectralArray,
                               const volDim* iVolDim);
}

#endif // SPECTRUM_H
