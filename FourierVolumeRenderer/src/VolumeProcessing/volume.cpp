#include "volume.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

char*** Volume::allocCubeVolume_char(const int size_X,
                                     const int size_Y,
                                     const int size_Z)
{
    LOG();

    INFO("Allocating BYTE CUBE volume : "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ));

    char*** cubeVolume;
    cubeVolume = (char***) malloc(size_X * sizeof(char**));
    for (int y = 0; y < size_X; y++)
    {
        cubeVolume[y] = (char**) malloc (size_Y* sizeof(char*));
        for (int x = 0; x < size_Y; x++)
        {
            cubeVolume[y][x] = (char*) malloc(size_Z * sizeof(char));
        }
    }

    INFO("BYTE CUBE volume allocation DONE");

    return cubeVolume;
}

float*** Volume::allocCubeVolume_float(const int size_X,
                                       const int size_Y,
                                       const int size_Z)
{
    INFO("Allocating FLOAT32 CUBE volume : "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ));

    float*** cubeVolume;
    cubeVolume = (float***) malloc(size_X * sizeof(float**));
    for (int y = 0; y < size_X; y++)
    {
        cubeVolume[y] = (float**) malloc (size_Y* sizeof(float*));
        for (int x = 0; x < size_Y; x++)
        {
            cubeVolume[y][x] = (float*) malloc(size_Z * sizeof(float));
        }
    }

    INFO("FLOAT32 CUBE volume allocation DONE");

    return cubeVolume;
}

void Volume::packFlatVolume(volume* iVolume,
                            char*** cubeVolume)
{
    INFO("Packing CUBE volume in FLAT array : "
         + STRG( "[" ) + ITS( iVolume->sizeX ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeY ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeZ ) + STRG( "]" ));

    int voxel = 0;
    for (int i = 0; i < iVolume->sizeX; i++)
    {
        for (int j = 0; j < iVolume->sizeY; j++)
        {
            for (int k = 0; k < iVolume->sizeZ; k++)
            {
                iVolume->ptrVol_char[voxel] = cubeVolume[i][j][k];
                voxel++;
            }
        }
    }

    INFO("Packing CUBE volume in FLAT array DONE");
}

void Volume::packCubeVolume(char*** cubeVolume, volume* iVolume)
{
    INFO("Packing FLAT volume in CUBE array : "
         + STRG( "[" ) + ITS( iVolume->sizeX ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeY ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeZ ) + STRG( "]" ));

    int voxel = 0;
    for (int i = 0; i < iVolume->sizeX; i++)
    {
        for (int j = 0; j < iVolume->sizeY; j++)
        {
            for (int k = 0; k < iVolume->sizeZ; k++)
            {
                cubeVolume[i][j][k] = iVolume->ptrVol_char[voxel];
                voxel++;
            }
        }
    }

    INFO("Packing FLAT volume in CUBE array DONE");
}

void Volume::extractSubVolume(char*** originalCubeVol,
                              char*** finalCubeVol,
                              const subVolDim* iSubVolDim)
{
    /* @ Calculating the sub volume dimensions */
    const int finalSize_X = iSubVolDim->max_X - iSubVolDim->min_X;
    const int finalSize_Y = iSubVolDim->max_Y - iSubVolDim->min_Y;
    const int finalSize_Z = iSubVolDim->max_Z - iSubVolDim->min_Z;

    INFO("Extracting SUB-VOLUME with dimensions : "
         + STRG( "[" ) + ITS( finalSize_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( finalSize_Y ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( finalSize_Y ) + STRG( "]" ));

    for (int i = 0; i < finalSize_X; i++)
    {
        for (int j = 0; j < finalSize_Y; j++)
        {
            for (int k = 0; k < finalSize_Z; k++)
            {
                finalCubeVol[i][j][k] = originalCubeVol
                        [(iSubVolDim->min_X) + i]
                        [(iSubVolDim->min_Y) + j]
                        [(iSubVolDim->min_Z) + k];
            }
        }
    }

    INFO("Extracting SUB-VOLUME DONE");
}

volume* Volume::extractFinalVolume(volume* iOriginaVol,
                                   const subVolDim* iSubVol)
{
     INFO("Extracting SUB-VOLUME to be processed in a SINGLE thread");

     INFO("Allocating CUBE array for the original volume : "
          + STRG( "[" ) + ITS( iOriginaVol->sizeX ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( iOriginaVol->sizeY ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( iOriginaVol->sizeZ ) + STRG( "]" ));

    /* @ Allocating cube array for the original volume */
    char*** originalCubeVol = Volume::allocCubeVolume_char
            (iOriginaVol->sizeX, iOriginaVol->sizeY, iOriginaVol->sizeZ);

    INFO("Packing the FLAt array in the CUBE ");

    /* @ Packing the original flat volume in the cube */
    Volume::packCubeVolume(originalCubeVol, iOriginaVol);

    /* @ Allocating the final cube volume "SUB-VOLUME" */
    volDim iVolDim;
    iVolDim.size_X = iSubVol->max_X - iSubVol->min_X;
    iVolDim.size_Y = iSubVol->max_Y - iSubVol->min_Y;
    iVolDim.size_Z = iSubVol->max_Z - iSubVol->min_Z;

    INFO("Allocating CUBE array for the final SUB-VOLUME: "
              + STRG( "[" ) + ITS( iVolDim.size_X ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolDim.size_Y ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolDim.size_Z ) + STRG( "]" ));

    char*** finalCubeVol = Volume::allocCubeVolume_char
            (iVolDim.size_X, iVolDim.size_Y, iVolDim.size_Z);

    INFO("Extractng the SUB-VOLUME");

    /* @ Extractig the SUB-VOLUME */
    Volume::extractSubVolume(originalCubeVol, finalCubeVol, iSubVol);

    /* @ Dellocating the original cube volume */
    FREE_MEM_3D_CHAR(originalCubeVol, iOriginaVol->sizeX, iOriginaVol->sizeY, iOriginaVol->sizeZ);

    INFO("Allocating FLAT volume for the SUB-VOLUME");

    /* @ Allocating the final flat volume */
    char* finalFlatVolume = (char*) malloc (sizeof(char) * (iVolDim.size_X)
                                            * (iVolDim.size_Y)
                                            * (iVolDim.size_Z));

    /* @ Packing the final cube volume in the flat array */
    Volume::packFlatVolume(iOriginaVol, finalCubeVol);

    INFO("Final volume extraction DONE");

    volume* finalVolume = (volume*) malloc (sizeof(volume));
    finalVolume->ptrVol_char = iOriginaVol->ptrVol_char;
    finalVolume->sizeX = 256;
    finalVolume->sizeY=256;
    finalVolume->sizeZ=256;
    finalVolume->sizeUni=256;


    return finalVolume;
}

volume* Volume::createFloatVolume(volume* iVolume_char)
{
    INFO("Creating FLAT FLOAT32 volume : "
              + STRG( "[" ) + ITS( iVolume_char->sizeX ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolume_char->sizeY ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolume_char->sizeZ ) + STRG( "]" ));

    /* @ Allocating float flat array */
    float* flatVol_float = (float*) malloc
            (sizeof(float*) * iVolume_char->sizeX * iVolume_char->sizeY * iVolume_char->sizeZ);

    /* @ Type conversion */
    for (int i = 0; i < iVolume_char->sizeX * iVolume_char->sizeY * iVolume_char->sizeZ; i++)
        flatVol_float[i] = (float) (unsigned char) iVolume_char->ptrVol_char[i];

    /* @ linking the float array to the original volume structure */
    iVolume_char->ptrVol_float = flatVol_float;

    INFO("Freeing the BYTE volume");

    /* @ freeing the BYTE volume */
    free(iVolume_char->ptrVol_char);

    INFO("Creating FLAT FLOAT32 volume DONE");

    return iVolume_char;
}
