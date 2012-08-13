#ifndef WRAPPINGAROUND_H
#define WRAPPINGAROUND_H

/* @ System includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* @ FFTW includes */
#include <fftw3.h>

namespace WrappingAround
{
    /* @ Shifting the spatial volume */
    void WrapAroundVolume(float* eFlatVolume,
                          const int N);

    /* @ Shifting the spectral volume */
    void WrapAroundSpectrum(float* eFlatVolume,
                            fftwf_complex* eFlatVolume_complex,
                            const int N);

    /* @ Shifting the reconstructed projection image */
    void WrapAroundImage(float** eSquareImage_MAIN,
                         float** eSquareImage_TEMP,
                         float* eFlatImage,
                         const int N);
}

#endif // WRAPPINGAROUND_H
