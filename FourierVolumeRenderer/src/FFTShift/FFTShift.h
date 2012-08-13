#ifndef FFTSHIFT_H
#define FFTSHIFT_H

#include "shared.h"

namespace FFTShift
{
    float** FFT_Shift_2D(float** iArr, float** oArr, int N);
    float*** FFT_Shift_3D(float* Input, int N);

    float* Repack_2D(float** Input_2D, float* Input_1D, int N);
    float* Repack_3D(float*** Input_3D, float* Input_1D, int N);
}

#endif // FFTSHIFT_H
