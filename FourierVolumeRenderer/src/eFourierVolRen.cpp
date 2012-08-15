#include "eFourierVolRen.h"
#include "OpenGL/cOpenGL.h"
#include "SpectrumProcessing/Spectrum.h"
#include "Loader/Loader.h"
#include "VolumeProcessing/volume.h"
#include "FFTShift/FFTShift.h"
#include "WrappingAround/WrappingAround.h"
#include "OpenGL/DisplayList.h"
#include "SliceProcessing/Slice.h"
#include "FFTShift/FFTShift.h"
#include "RenderingLoop/RenderingLoop.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

GLuint eVolumeTexture_ID;
GLuint eSliceTexture_ID;
GLuint eImageTexture_ID;
GLuint eFBO_ID;
sliceDim eSliceDim_Glob;

/* @ NOTE: You can ONLY rotate the cube around +/- X, Y, Z by 90 degrees */
float eXRot_Glob            = 0;
float eYRot_Glob            = -90;
float eZRot_Glob            = 0;

float eZoomLevel_Glob       = 2;
float eSliceTrans_Glob      = 0;
float eSliceUniSize_Glob    = 0;
float eNormValue_Glob       = 150;

/* @ Enabling volume decomposition */
bool ENABLE_BREAK = 1;

using namespace Magick;

Image*  GetSpectrumSlice()
{
    Image *eImage;

    eImage = RenderingLoop::run(eXRot_Glob,
                                eYRot_Glob,
                                eZRot_Glob,
                                eSliceTrans_Glob, 1,
                                eSliceDim_Glob.size_X, eSliceDim_Glob.size_Y,
                                &eSliceTexture_ID,
                                &eVolumeTexture_ID,
                                eFBO_ID,
                                &eImageTexture_ID);

    return eImage;
}


volume* eFourierVolRen::loadingVol(char* iVolPath)
{
    LOG();

    INFO("LOADING VOLUME");

    /* @ Loading the input volume dataset from the path */
    volume* eVolume_char = Loader::loadVolume(iVolPath);

    return eVolume_char;
}

void eFourierVolRen::initContexts(int argc, char** argv)
{
    LOG();

    INFO("INITIALIZING CONTEXT");

    OpenGL::initOpenGLContext(argc, argv);

    /* @ OpenGL initialization */
    OpenGL::initOpenGL();

}

volume* eFourierVolRen::decomposeVolume(volume* iVolume, int iBrickIndex)
{
    /***********************
      FACE ONE (UPPER FACE)
       ___________________
      |         |         |
      |    1    |    3    |
      |_________|_________|
      |         |         |
      |    2    |    4    |
      |_________|_________|


      FACE TWO (LOWER FACE)
       ___________________
      |         |         |
      |    5    |    7    |
      |_________|_________|
      |         |         |
      |    6    |    8    |
      |_________|_________|

    ************************/

    LOG();

    INFO("BREAKING STAGE - VOLUME DECOMPOSITION");

    subVolDim eSubVolDim;
    if (iBrickIndex == 0)
    {
        eSubVolDim.min_X = 0;
        eSubVolDim.min_Y = 0;
        eSubVolDim.min_Z = 0;
        eSubVolDim.max_X = (iVolume->sizeX / 2);
        eSubVolDim.max_Y = (iVolume->sizeY / 2);
        eSubVolDim.max_Z = (iVolume->sizeZ / 2);
    }
    else if (iBrickIndex == 1)
    {
        eSubVolDim.min_X = 0;
        eSubVolDim.min_Y = 0;
        eSubVolDim.min_Z = (iVolume->sizeZ / 2);
        eSubVolDim.max_X = (iVolume->sizeX / 2);
        eSubVolDim.max_Y = (iVolume->sizeY / 2);
        eSubVolDim.max_Z = (iVolume->sizeZ);
    }
    else if (iBrickIndex == 2)
    {
        eSubVolDim.min_X = 0;
        eSubVolDim.min_Y = (iVolume->sizeX / 2);
        eSubVolDim.min_Z = 0;
        eSubVolDim.max_X = (iVolume->sizeX / 2);
        eSubVolDim.max_Y = (iVolume->sizeY);
        eSubVolDim.max_Z = (iVolume->sizeZ / 2);
    }
    else if (iBrickIndex == 3)
    {
        eSubVolDim.min_X = 0;
        eSubVolDim.min_Y = (iVolume->sizeX / 2);
        eSubVolDim.min_Z = (iVolume->sizeZ / 2);
        eSubVolDim.max_X = (iVolume->sizeX / 2);
        eSubVolDim.max_Y = (iVolume->sizeY);
        eSubVolDim.max_Z = (iVolume->sizeZ);
    }
    else if (iBrickIndex == 4)
    {
        eSubVolDim.min_X = (iVolume->sizeX / 2);
        eSubVolDim.min_Y = 0;
        eSubVolDim.min_Z = 0;
        eSubVolDim.max_X = (iVolume->sizeX);
        eSubVolDim.max_Y = (iVolume->sizeY / 2);
        eSubVolDim.max_Z = (iVolume->sizeZ / 2);
    }
    else if (iBrickIndex == 5)
    {
        eSubVolDim.min_X = (iVolume->sizeX / 2);;
        eSubVolDim.min_Y = 0;
        eSubVolDim.min_Z = (iVolume->sizeZ / 2);
        eSubVolDim.max_X = (iVolume->sizeX);
        eSubVolDim.max_Y = (iVolume->sizeY / 2);
        eSubVolDim.max_Z = (iVolume->sizeZ);
    }
    else if (iBrickIndex == 6)
    {
        eSubVolDim.min_X = (iVolume->sizeX / 2);;
        eSubVolDim.min_Y = (iVolume->sizeX / 2);
        eSubVolDim.min_Z = 0;
        eSubVolDim.max_X = (iVolume->sizeX);
        eSubVolDim.max_Y = (iVolume->sizeY);
        eSubVolDim.max_Z = (iVolume->sizeZ / 2);
    }
    else if (iBrickIndex == 7)
    {
        eSubVolDim.min_X = (iVolume->sizeX / 2);;
        eSubVolDim.min_Y = (iVolume->sizeX / 2);
        eSubVolDim.min_Z = (iVolume->sizeZ / 2);
        eSubVolDim.max_X = (iVolume->sizeX);
        eSubVolDim.max_Y = (iVolume->sizeY);
        eSubVolDim.max_Z = (iVolume->sizeZ);
    }
    else if (iBrickIndex == -1)
    {
        eSubVolDim.min_X = 0;
        eSubVolDim.min_Y = 0;
        eSubVolDim.min_Z = 0;
        eSubVolDim.max_X = (iVolume->sizeX);
        eSubVolDim.max_Y = (iVolume->sizeY);
        eSubVolDim.max_Z = (iVolume->sizeZ);
    }
    else
    {
        INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
        EXIT(0);
    }

    volume* finalVolume = Volume::extractFinalVolume(iVolume, &eSubVolDim);

    return finalVolume;
}

sliceDim* eFourierVolRen::preProcess(volume* iVolume)
{
    LOG();

    INFO("PREPROCESSING STAGE");

    /* @ Final volume & slice size */
    sliceDim* eSliceDim;
    eSliceDim = (sliceDim*) malloc (sizeof(sliceDim));

    volDim* eVolDim;
    eVolDim = (volDim*) malloc (sizeof(volDim));

    /* @ Creating the float volume */
    iVolume = Volume::createFloatVolume(iVolume);

    /* @ Setting the final volume & slice dimensions */
    eSliceDim->size_X = iVolume->sizeX;
    eSliceDim->size_Y = iVolume->sizeY;
    eVolDim->size_X = iVolume->sizeX;
    eVolDim->size_Y = iVolume->sizeY;
    eVolDim->size_Z = iVolume->sizeZ;

    if (eSliceDim->size_X == eSliceDim->size_Y)
    {
        INFO("UNIFIED DIMENSIONS " + ITS(eSliceDim->size_X));

        /* Getting the slice size for the adjusting the view port */
        eSliceUniSize_Glob = eSliceDim->size_X;
    }

    /* @ Wrapping around the spatial volume */
    WrappingAround::WrapAroundVolume
            (iVolume->ptrVol_float, iVolume->sizeUni);

    /* @ Creating the complex spectral volume */
    fftwf_complex* eVolumeData_complex =
            Spectrum::createSpectrum(iVolume);

    /* @ Wrapping around the spectral volume */
    WrappingAround::WrapAroundSpectrum
            (iVolume->ptrVol_float, eVolumeData_complex, iVolume->sizeUni);

    /* @ Packing the spectrum volume data into texture
     * array to be sent to OpenGL */
    float* eSpectrumTexture = Spectrum::packingSpectrumTexture(eVolumeData_complex, eVolDim);

    /* @ Uploading the spectrum to the GPU texture */
    Spectrum::uploadSpectrumTexture(&eVolumeTexture_ID, eSpectrumTexture, eVolDim);

    return eSliceDim;
}

Image* eFourierVolRen::rendering(const int iSliceWidth, const int iSliceHeight)
{
    LOG();

    INFO("RENDERING LOOP");

    /* @ Creating the projection slice texture & binding it */
    Slice::createSliceTexture(iSliceWidth, iSliceHeight, &eSliceTexture_ID);

    /* @ Prepare the FBO & attaching the slice texture to it */
    OpenGL::prepareFBO(&eFBO_ID, &eSliceTexture_ID);

    /* @ Preparing rendering arrays */
    RenderingLoop::prepareRenderingArray(iSliceWidth, iSliceHeight);

    eSliceDim_Glob.size_X = iSliceWidth;
    eSliceDim_Glob.size_Y = iSliceHeight;


    /* @ Rendering loop */
    Image* eImage = GetSpectrumSlice();

    // glutMainLoop();

    return eImage;
}

void eFourierVolRen::setBrick(Image* iTile,
                              compositionImages* iImageList,
                              int iXOffset, int iYOffset,
                              int iTileWidth, int iTileHeight, int iH)
{
    if (iH == 1)
    {
        for (int i = 0; i < iTileWidth; i++)
        {
            for (int j = 0; j < iTileHeight; j++)
            {
               ColorGray gsColor(iTile->pixelColor(i, j));
               iImageList->image_H1->pixelColor(i + iXOffset, j + iYOffset, gsColor);
            }
        }
    }
    else if (iH == 2)
    {
        for (int i = 0; i < iTileWidth; i++)
        {
            for (int j = 0; j < iTileHeight; j++)
            {
               ColorGray gsColor(iTile->pixelColor(i, j));
               iImageList->image_H2->pixelColor(i + iXOffset, j + iYOffset, gsColor);
            }
        }
    }
}

void eFourierVolRen::addTileToFinalImage(int iFinalSliceWidth,
                                         int iFinalSliceHeight,
                                         Image* iTile,
                                         compositionImages* iImageList,
                                         int iBrickIndex)
{
    INFO("Adding TILES to the final image : " + ITS(iBrickIndex));

    INFO("Final image geometry : " +
         CATS("[") + ITS(iFinalSliceWidth) + CATS("]") + CATS(" x ") +
         CATS("[") + ITS(iFinalSliceHeight) + CATS("]"));

    const int eTileWidth = iFinalSliceWidth / 2;
    const int eTileHeight = iFinalSliceHeight / 2;

    if (eXRot_Glob == 0 && eYRot_Glob == 0 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 4)
             setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 90 && eYRot_Glob == 0 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 0 && eYRot_Glob == 90 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 0 && eYRot_Glob == 90 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 0 && eYRot_Glob == 0 && eZRot_Glob == 90)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == -90 && eYRot_Glob == 0 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 0 && eYRot_Glob == -90 && eZRot_Glob == 0)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
             setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }
    else if (eXRot_Glob == 0 && eYRot_Glob == 0 && eZRot_Glob == -90)
    {
        if (iBrickIndex == 0)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 1)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 2)
            setBrick(iTile, iImageList, eTileWidth, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 3)
            setBrick(iTile, iImageList, eTileWidth, eTileHeight, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 4)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 5)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 1);
        else if (iBrickIndex == 6)
            setBrick(iTile, iImageList, 0, 0, eTileWidth, eTileHeight, 2);
        else if (iBrickIndex == 7)
            setBrick(iTile, iImageList, 0, eTileHeight, eTileWidth, eTileHeight, 2);
        else
        {
            INFO("Brick index OUT OF RANGE - QUITTING SAFELY");
            EXIT(0);
        }
    }


     INFO("Adding TILES to the final image DONE ");
}

compositionImages* eFourierVolRen::createFinalImage(int iFinalSliceWidth, int iFinalSliceHeight)
{
    INFO("Creating IMAGE LIST");

    INFO("Final image geometry : " +
         CATS("[") + ITS(iFinalSliceWidth) + CATS("]") + CATS(" x ") +
         CATS("[") + ITS(iFinalSliceHeight) + CATS("]"));

    /* Define image geometry */
    Geometry eGeom_H, eGeom_F;
    eGeom_H.width(iFinalSliceWidth);
    eGeom_H.height(iFinalSliceHeight);
    eGeom_F.width(iFinalSliceWidth);
    eGeom_F.height(iFinalSliceHeight);

    compositionImages* imageList;
    imageList = (compositionImages*) malloc (sizeof(compositionImages));

    /* Allocate the images */
    Image *image_H1, *image_H2, *image_Final;
    image_H1 = new Image();
    image_H2 = new Image();
    image_Final = new Image();


    image_Final->quality(00);
    image_Final->defineSet("png:color-type", "6");
    image_Final->defineSet("png:bit-depth", "32");
    image_Final->defineSet("png:format", "png32");

    image_H1->type(GrayscaleType);
    image_H2->type(GrayscaleType);
    image_Final->type(GrayscaleType);

    image_H1->size(eGeom_H);
    image_H2->size(eGeom_H);
    image_Final->size(eGeom_F);

    /* @ Initializing the final image to BLACK */
    for (int i = 0; i < iFinalSliceWidth; i++)
    {
        for (int j = 0; j < iFinalSliceHeight; j++)
        {
           ColorGray gsColor(0);
           image_H1->pixelColor(i, j, gsColor);
           image_H2->pixelColor(i, j, gsColor);
        }
    }

    imageList->image_H1 = image_H1;
    imageList->image_H2 = image_H2;
    imageList->image_Final = image_Final;

    return imageList;
}

void composeFinalProjection(compositionImages* iImageList, int iFinalSliceWidth, int iFinalSliceHeight)
{
    INFO("Composing final image ");

    for (int i = 0; i < iFinalSliceWidth; i++)
    {
        for (int j = 0; j < iFinalSliceHeight; j++)
        {
           float eIntensityValue = 0;
           ColorGray gsColor(eIntensityValue);
           iImageList->image_Final->pixelColor(i, j, gsColor);
        }
    }

    iImageList->image_Final->composite(*iImageList->image_H1, 0, 0, BlendCompositeOp);
    iImageList->image_Final->composite(*iImageList->image_H2, 0, 0, BlendCompositeOp);

    INFO("Composing final image DONE ");
}

void eFourierVolRen::writeFinalImageToDisk(compositionImages* iImageList)
{
    iImageList->image_H1->write( "Half_1.png" );
    iImageList->image_H2->write( "Half_2.png" );
    iImageList->image_Final->write("FinalProjection.png");
}

void eFourierVolRen::run(int argc, char** argv, char* iVolPath)
{
    LOG();

    INFO("eFourierVolRen :: RUNNING");

    Image* eImage;
    Image* eTile;

    /* @ Loading volume */
    volume* eVol = eFourierVolRen::loadingVol(iVolPath);

    /* @ Initializing contextx */
    eFourierVolRen::initContexts(argc, argv);

    compositionImages* eImageList;

    /* Creating the final image */
    eImageList = eFourierVolRen::createFinalImage(eVol->sizeX, eVol->sizeY);

    if (ENABLE_BREAK)
    {
        /* Decomposing the volume into 8 bricks */
        for (int eBrickIndex = 0; eBrickIndex < 8; eBrickIndex++)
        {
            /* @ Decompose volume */
            volume* eVolBrick = decomposeVolume(eVol, eBrickIndex);

            /* @ Preprocessnig stage */
            sliceDim* eSliceDim = eFourierVolRen::preProcess(eVolBrick);

            /* @ Rendering loop */
            eTile = eFourierVolRen::rendering(eSliceDim->size_X, eSliceDim->size_Y);

            addTileToFinalImage(eVol->sizeX, eVol->sizeY, eTile, eImageList, eBrickIndex);

        }

        /*  Composing the final projection */
        composeFinalProjection(eImageList, eVol->sizeX, eVol->sizeY);

        /* @ Writing the final projections to the file system */
        writeFinalImageToDisk(eImageList);
    }
    else
    {
        /* @ Decompose volume */
        volume* eVolBrick = decomposeVolume(eVol, -1);

        /* @ Preprocessnig stage */
        sliceDim* eSliceDim = eFourierVolRen::preProcess(eVolBrick);

        /* @ Rendering loop */
        eTile = eFourierVolRen::rendering(eSliceDim->size_X, eSliceDim->size_Y);


    }
}
