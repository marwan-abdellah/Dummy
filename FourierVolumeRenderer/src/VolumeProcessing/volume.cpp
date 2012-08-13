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

void Volume::packFlatVolume(char* flatVolume,
                            char*** cubeVolume,
                            const volDim* iVolDim)
{
    INFO("Packing CUBE volume in FLAT array : "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Y ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Z ) + STRG( "]" ));

    int voxel = 0;
    for (int i = 0; i < iVolDim->size_X; i++)
    {
        for (int j = 0; j < iVolDim->size_Y; j++)
        {
            for (int k = 0; k < iVolDim->size_Z; k++)
            {
                flatVolume[voxel] = cubeVolume[i][j][k];
                voxel++;
            }
        }
    }

    INFO("Packing CUBE volume in FLAT array DONE");
}

void Volume::packCubeVolume(char*** cubeVolume,
                            char* flatVolume,
                            const volDim* iVolDim)
{
    INFO("Packing FLAT volume in CUBE array : "
         + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Y ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolDim->size_Z ) + STRG( "]" ));

    int voxel = 0;
    for (int i = 0; i < iVolDim->size_X; i++)
    {
        for (int j = 0; j < iVolDim->size_Y; j++)
        {
            for (int k = 0; k < iVolDim->size_Z; k++)
            {
                cubeVolume[i][j][k] = flatVolume[voxel];
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

char* Volume::extractFinalVolume(char* originalFlatVol,
                                 const volDim* originalDim,
                                 const subVolDim* iSubVol)
{
     INFO("Extracting final volume to be processed in a single thread");

     INFO("Allocating CUBE array for the original volume : "
          + STRG( "[" ) + ITS( originalDim->size_X ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( originalDim->size_Y ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( originalDim->size_Z ) + STRG( "]" ));

    /* @ Allocating cube array for the original volume */
    char*** originalCubeVol = Volume::allocCubeVolume_char
            (originalDim->size_X, originalDim->size_Y, originalDim->size_Z);

    INFO("Packing the FLAt array in the CUBE ");

    /* @ Packing the original flat volume in the cube */
    Volume::packCubeVolume(originalCubeVol, originalFlatVol, originalDim);

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
    FREE_MEM_3D_CHAR(originalCubeVol, originalDim->size_X, originalDim->size_Y, originalDim->size_Z);

    INFO("Allocating FLAT volume for the SUB-VOLUME");

    /* @ Allocating the final flat volume */
    char* finalFlatVolume = (char*) malloc (sizeof(char) * (iVolDim.size_X)
                                            * (iVolDim.size_Y)
                                            * (iVolDim.size_Z));

    /* @ Packing the final cube volume in the flat array */
    Volume::packFlatVolume(finalFlatVolume, finalCubeVol, &iVolDim);

    INFO("Final volume extraction DONE");

    return finalFlatVolume;
}

float* Volume::createFloatVolume(char* flatVol_char, const volDim* iVolDim)
{
    INFO("Creating FLAT FLOAT32 volume : "
              + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolDim->size_Y ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolDim->size_Z ) + STRG( "]" ));

    float* flatVol_float = (float*) malloc
            (sizeof(float*) * iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z);

    for (int i = 0; i < iVolDim->size_X * iVolDim->size_Y * iVolDim->size_Z; i++)
        flatVol_float[i] = (float) (unsigned char) flatVol_char[i];

    INFO("Creating FLAT FLOAT32 volume DONE");

    return flatVol_float;
}
