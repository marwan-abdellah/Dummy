#include "WrappingAround.h"



// 3D Wrapping Around for Space Data
void WrappingAround::WrapAroundVolume(float*** Vol_3D, float* VolumeDataFloat, int N)
{
    printf("Wrapping Around Volume Data ... \n");

    Vol_3D = FFTShift::FFT_Shift_3D(VolumeDataFloat, N);
    VolumeDataFloat = FFTShift::Repack_3D(Vol_3D, VolumeDataFloat, N);

    printf("	Wrapping Around Volume Data Done Successfully \n\n");
}

// 3D Wrapping Around the Spectrum
void WrappingAround::WrapAroundSpectrum(float*** Vol_3D, float* VolumeDataFloat, fftwf_complex* VolumeArrayComplex, int N)
{
    printf("Wrapping Around Spectrum Data ... \n");
    printf("	Real Part ... \n");

    for (int i = 0; i < N*N*N; i++)
        VolumeDataFloat[i] =  VolumeArrayComplex[i][0];

   Vol_3D = FFTShift::FFT_Shift_3D(VolumeDataFloat, N);
    VolumeDataFloat = FFTShift::Repack_3D(Vol_3D, VolumeDataFloat, N);

    for (int i = 0; i < N*N*N; i++)
       VolumeArrayComplex[i][0] = VolumeDataFloat[i];

    printf("	Imaginary Part ... \n");

    for (int i = 0; i < N*N*N; i++)
        VolumeDataFloat[i] =  VolumeArrayComplex[i][1];

    Vol_3D = FFTShift::FFT_Shift_3D(VolumeDataFloat, N);
    VolumeDataFloat = FFTShift::Repack_3D(Vol_3D, VolumeDataFloat, N);

    for (int i = 0; i < N*N*N; i++)
        VolumeArrayComplex[i][1] = VolumeDataFloat[i];

    printf("	Wrapping Around Spectrum Data Done Successfully \n\n");
}
