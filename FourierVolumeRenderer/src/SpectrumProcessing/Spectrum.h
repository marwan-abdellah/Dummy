#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "shared.h"
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

// FFTW Includes
#ifndef WIN32
    #include <fftw3.h>
#else
    #include "fftw3.h"
#endif

namespace Spectrum
{
    fftwf_complex* createSpectrum(float* spatialVol, volDim* iVolDim);
    float* packingSpectrumTexture(fftwf_complex* spectralVol, volDim* iVolDim);
    void UploadSpectrumTexture(GLuint* spectrumTexID, float* spectrum, volDim* iVolDim);
}

#endif // SPECTRUM_H
