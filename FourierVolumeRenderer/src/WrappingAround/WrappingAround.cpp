#include "WrappingAround.h"
#include "FFTShift/FFTShift.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

void WrappingAround::WrapAroundVolume(float* eFlatVolume,
                                      const int N)
{
    INFO("Wrapping-around SPATIAL volume with unified dimensions "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ) + " x "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ) + " x "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ));

    float*** eCubeVolume;

   eCubeVolume = FFTShift::FFT_Shift_3D(eFlatVolume, N);
   eFlatVolume = FFTShift::Repack_3D(eCubeVolume, eFlatVolume, N);

    //eFlatVolume = FFTShift::flatFFT_Shift_3D(eFlatVolume, N);

    INFO("Wrapping-around SPATIAL volume DONE");
}

void WrappingAround::WrapAroundSpectrum(float* eFlatVolume,
                                        fftwf_complex* eFlatVolume_complex,
                                        const int N)
{
    INFO("Wrapping-around SPECTRAL volume with unified dimensions "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ) + " x "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ) + " x "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ));

    float*** eCubeVolume;

    INFO("Real Part ...");

    for (int i = 0; i < (N * N * N); i++)
        eFlatVolume[i] =  eFlatVolume_complex[i][0];

    eCubeVolume = FFTShift::FFT_Shift_3D(eFlatVolume, N);
    eFlatVolume = FFTShift::Repack_3D(eCubeVolume, eFlatVolume, N);

    for (int i = 0; i < (N * N * N); i++)
        eFlatVolume_complex[i][0] = eFlatVolume[i];

    INFO("Imaginary Part ...");

    for (int i = 0; i < N*N*N; i++)
        eFlatVolume[i] =  eFlatVolume_complex[i][1];

    eCubeVolume = FFTShift::FFT_Shift_3D(eFlatVolume, N);
    eFlatVolume = FFTShift::Repack_3D(eCubeVolume, eFlatVolume, N);

    for (int i = 0; i < N*N*N; i++)
        eFlatVolume_complex[i][1] = eFlatVolume[i];

    INFO("Wrapping-around SPECTRAL volume DONE");
}

void WrappingAround::WrapAroundImage(float** eSquareImage_MAIN,
                                     float** eSquareImage_TEMP,
                                     float* eFlatImage,
                                     const int N)
{

    INFO("Wrapping-around PROJECTION image with unified dimensions "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ) + " x "
            + STRG( "[" ) + ITS( N ) + STRG( "]" ));

    eSquareImage_MAIN = FFTShift::FFT_Shift_2D(eSquareImage_TEMP, eSquareImage_MAIN, N);
    eFlatImage = FFTShift::Repack_2D(eSquareImage_MAIN, eFlatImage, N);

    INFO("Wrapping-around PROJECTION image DONE");
}

