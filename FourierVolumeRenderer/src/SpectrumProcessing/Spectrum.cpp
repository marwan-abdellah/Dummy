#include "Spectrum.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

fftwf_complex* Spectrum::createSpectrum(volume* iSpectralVolume)
{
    LOG();

    INFO("Creating COMPLEX SPECTRUM : "
         + STRG( "[" ) + ITS( iSpectralVolume->sizeX ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iSpectralVolume->sizeY ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iSpectralVolume->sizeZ ) + STRG( "]" ));

    /* @ Allocatig spectral volume */
    fftwf_complex* eSpectralVolume_complex = (fftwf_complex*) fftwf_malloc
            (iSpectralVolume->sizeX * iSpectralVolume->sizeY * iSpectralVolume->sizeZ
             * sizeof(fftwf_complex));

    /* @ Packing complex array in interleaved manner */
    for (int i = 0; i < (iSpectralVolume->sizeX * iSpectralVolume->sizeY * iSpectralVolume->sizeZ); i++)
    {
        eSpectralVolume_complex[i][0] = iSpectralVolume->ptrVol_float[i];
        eSpectralVolume_complex[i][1] = iSpectralVolume->ptrVol_float[i];

    }

    INFO("3D FFT");

    /* @ 3D FFT */
    fftwf_plan eFFTPlan = fftwf_plan_dft_3d(
                            iSpectralVolume->sizeX,
                            iSpectralVolume->sizeY,
                            iSpectralVolume->sizeZ,
                            eSpectralVolume_complex,
                            eSpectralVolume_complex,
                            FFTW_FORWARD,
                            FFTW_ESTIMATE);
    /* @ executing the FFT plan */
    fftwf_execute(eFFTPlan);

    fftwf_destroy_plan(eFFTPlan);

    INFO("Creating COMPLEX SPECTRUM DONE");

    return eSpectralVolume_complex;
}


//  Packing 3D Texture with 2 Components, Real+Imaginary
float* Spectrum::packingSpectrumTexture(const fftwf_complex* iSpectralVolume,
                                        const volDim* iVolDim)
{
    INFO("Packing spectral CPU texture : "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Y ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Z ) + STRG( "]" ));

    const int eVolSize = iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z;

    /* @ Allocating the CPU spectral texture array */
    float* eSpectralTexture =  (float*) malloc (eVolSize * 2 * sizeof(float));
    /* @ Filling the CPU spectral texture */
    int ctr = 0;
    for (int i = 0; i < (eVolSize * 2); i += 2)
    {
        eSpectralTexture[i]		= iSpectralVolume[ctr][0];
        eSpectralTexture[i + 1]	= iSpectralVolume[ctr][1];
        ctr++;
    }

    INFO("Packing spectral CPU texture DONE");

    return eSpectralTexture;
}

// Send Texture Volume to the GPU Texture Memory
void Spectrum::uploadSpectrumTexture(GLuint* iSpectralTexture_ID,
                                     const float* iSpectralArray,
                                     const volDim* iVolDim)
{
    INFO("Creating, binding & uploading GPU spectral texture : "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ));

    /* @ 3D Texture creation & binding */
    glGenTextures(1, iSpectralTexture_ID);
    glBindTexture(GL_TEXTURE_3D, *iSpectralTexture_ID);

    /* @ Texture parameters */
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    /* @ For automatic texture coordinate generation */
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);

    INFO("Transfering the data to the GPU Memory");

    /* @ Uplading the texture to the GPU */
    glTexImage3D(GL_TEXTURE_3D, 0, RG32F,
                 iVolDim->size_X,
                 iVolDim->size_Y,
                 iVolDim->size_Z,
                 0, RG, GL_FLOAT,  iSpectralArray);

    INFO("Creating, binding & uploading GPU spectral texture DONE");

}
