#ifndef WRAPPINGAROUND_H
#define WRAPPINGAROUND_H

#include "FFTShift/FFTShift.h"
#include <fftw3.h>

namespace WrappingAround
{
    void WrapAroundVolume(float*** Vol_3D, float* VolumeDataFloat, int N);
    void WrapAroundSpectrum(float*** Vol_3D, float* VolumeDataFloat, fftwf_complex* VolumeArrayComplex, int N);
}

#endif // WRAPPINGAROUND_H
