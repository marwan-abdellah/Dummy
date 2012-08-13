#include "Spectrum.h"
#include "shared.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>


fftwf_complex* Spectrum::createSpectrum(float* spatialVol, volDim* iVolDim)
{
    printf("Creating Complex Spectrum ... \n");

    fftwf_complex* spectralVol =
            (fftwf_complex*) fftwf_malloc (iVolDim->size_X *
                                           iVolDim->size_Y *
                                           iVolDim->size_Z *
                                           sizeof(fftwf_complex));


    // Packing Complex Array
    printf("	Packing Volume Array for Single Precision Format ... \n");
    for (int i = 0; i < iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z ; i++)
    {
        spectralVol[i][0] = spatialVol[i];
        spectralVol[i][1] = spatialVol[i];

    }
    printf("	Packing Done Successfully \n\n");

    // 3D Forward Fourier Transforming Data
    printf("	Executing 3D Forward FFT  ... \n");

    fftwf_plan fftPlan = fftwf_plan_dft_3d(iVolDim->size_X,
                                  iVolDim->size_Y,
                                  iVolDim->size_Z,
                                  spectralVol,
                                  spectralVol,
                                  FFTW_FORWARD,
                                  FFTW_ESTIMATE);
    fftwf_execute(fftPlan);

    printf("	3D Forward FFT Done Successfully  \n\n");

    return spectralVol;
}


//  Packing 3D Texture with 2 Components, Real+Imaginary
float* Spectrum::packingSpectrumTexture(fftwf_complex* spectralVol, volDim* iVolDim)
{
    // Allocate Complex Texture Array to be Sent to the GPU
    // mTextureArray = (float*) malloc (mVolumeSize * 2 * sizeof(float));

    float* spectrumTexture =  (float*)
            malloc (2 * sizeof(float) * iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z);

    printf("Packing Spectrum Into Texture ... \n");

    int ctr = 0;
    for (int i = 0; i < (iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z * 2); i += 2)
    {
        spectrumTexture[i]		= spectralVol[ctr][0];
        spectrumTexture[i + 1]	= spectralVol[ctr][1];
        ctr++;
    }

    printf("	Packing Spectrum Into Texture Done Successfully \n\n");

    return spectrumTexture;
}

// Send Texture Volume to the GPU Texture Memory
void Spectrum::UploadSpectrumTexture(GLuint* spectrumTexID, float* spectrum, volDim* iVolDim)
{
    printf("Creating & Binding Spectrum Texture To GPU ... \n");

    // 3D Texture Creation & Binding
    glGenTextures(1, spectrumTexID);
    glBindTexture(GL_TEXTURE_3D, *spectrumTexID);

    // Parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // For Automatic Texture Coordinate Generation
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);

    // Transfer Data to GPU Memory
    printf("	Transfer Data to GPU Memory ... \n");

    glTexImage3D(GL_TEXTURE_3D, 0, RG32F, iVolDim->size_X, iVolDim->size_Y, iVolDim->size_Z, 0, RG, GL_FLOAT,  spectrum);

    printf("	Transfering Data to GPU Memory Done Successfully \n\n");
}
