#include "volume.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"



// char*** originalCubeVol;
// char*** finalCubeVol;

int Volume::prepareVolumeArrays(volume* iOriginaVol, int iFinalDim)
{
   //  originalCubeVol = Volume::allocCubeVolume_char
      //   (iOriginaVol->sizeX, iOriginaVol->sizeY, iOriginaVol->sizeZ);

    /* @ Allocating the final cube volume "SUB-VOLUME" */
    // finalCubeVol = Volume::allocCubeVolume_char
       // (256, 256, 256);
}


int Volume::getUnifiedDimension(int iMaxDim)
{
    LOG();

    int eUnifiedDim = 0;

    if (iMaxDim <= 16) eUnifiedDim = 16;
    else if (iMaxDim <= 32) eUnifiedDim = 32;
    else if (iMaxDim <= 64) eUnifiedDim = 64;
    else if (iMaxDim <= 128) eUnifiedDim = 128;
    else if (iMaxDim <= 256) eUnifiedDim = 256;
    else if (iMaxDim <= 512) eUnifiedDim = 512;
    else if (iMaxDim <= 1024) eUnifiedDim = 1024;
    else if (iMaxDim <= 2048) eUnifiedDim = 2048;
    else if (iMaxDim <= 4096) eUnifiedDim = 4096;
    else if (iMaxDim <= 8192) eUnifiedDim = 8192;

    return eUnifiedDim;
}

char*** Volume::allocCubeVolume_char(const int size_X,
                                     const int size_Y,
                                     const int size_Z)
{
    LOG();

    INFO("Allocating BYTE CUBE volume : "
         + STRG( "[" ) + ITS( size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_Y ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( size_Z ) + STRG( "]" ));

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

void Volume::unifyVolumeDim(volume* iVolume)
{
    int eMaxDim = 0;
    if (iVolume->sizeX >= iVolume->sizeY && iVolume->sizeZ >= iVolume->sizeZ)
        eMaxDim = iVolume->sizeX;
    else if (iVolume->sizeY >= iVolume->sizeX && iVolume->sizeY >= iVolume->sizeZ)
        eMaxDim = iVolume->sizeY;
    else
        eMaxDim = iVolume->sizeZ;

    INFO("MAX DIMENSION : " +  ITS(eMaxDim));

    /* @ Adjusting the power of two dimension condition */
    const int eUnifiedDim = getUnifiedDimension(eMaxDim);

    INFO("FINAL UNIFIED VOLUME DIMENSION : " +  ITS(eUnifiedDim));

    /* Calculating the zero-padded area */
    int eX_Pad = (eUnifiedDim - iVolume->sizeX) / 2;
    int eY_Pad = (eUnifiedDim - iVolume->sizeY) / 2;
    int eZ_Pad = (eUnifiedDim - iVolume->sizeZ) / 2;

    INFO("Orginal volume dimensions : "
         + STRG( "[" ) + ITS( iVolume->sizeX ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeY ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( iVolume->sizeZ ) + STRG( "]" ));

    INFO("PADDING : "
         + STRG( "[" ) + ITS( eX_Pad ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( eY_Pad ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( eZ_Pad ) + STRG( "]" ));

    /* @ Creating a new flat array for the UNIFIED volume */
    char* eUnfiromFlatVolume = (char*) malloc
            (sizeof(char) * eUnifiedDim * eUnifiedDim * eUnifiedDim);

    // Setting the unified volume
    for (int i = 0; i < iVolume->sizeX; i++)
        for (int j = 0; j < iVolume->sizeY; j++)
            for(int k = 0; k < iVolume->sizeZ; k++)
            {
                int oldIndex =  (k * iVolume->sizeX * iVolume->sizeY) +
                                (j * iVolume->sizeX) + i;
                int newIndex =  ((k + eZ_Pad) * eUnifiedDim * eUnifiedDim) +
                                ((j + eY_Pad) * eUnifiedDim) + (i + eX_Pad);

                eUnfiromFlatVolume [newIndex] = iVolume->ptrVol_char[oldIndex];
            }

    /* @ Freeing the original input array */
    free(iVolume->ptrVol_char);

    /* @ Poiting to the new UNIFIED flat array */
    iVolume->ptrVol_char = eUnfiromFlatVolume;

    /* @ Adjusting the new volume parameters */
    iVolume->sizeX = eUnifiedDim;
    iVolume->sizeY = eUnifiedDim;
    iVolume->sizeZ = eUnifiedDim;
    iVolume->sizeUni = eUnifiedDim;
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

void extractSubVolume_Flat(char* originalCubeVol,
                              char* finalCubeVol,
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
                {
int oldIndex = ((k + iSubVolDim->min_Z) * 512 * 512) + ((j +iSubVolDim->min_Y ) * 512) + (i + iSubVolDim->min_X);
int newIndex = ((k) * finalSize_X * finalSize_Y) + ((j) * finalSize_X) + (i);

                    finalCubeVol [newIndex] = originalCubeVol[oldIndex];
                }
            }
        }
    }

    INFO("Extracting SUB-VOLUME DONE");
}

volume* Volume::extractFinalVolume(volume* iOriginaVol,
                                   const subVolDim* iSubVolDim)
{
     INFO("Extracting SUB-VOLUME to be processed in a SINGLE thread");

     INFO("Allocating CUBE array for the original volume : "
          + STRG( "[" ) + ITS( iOriginaVol->sizeX ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( iOriginaVol->sizeY ) + STRG( "]" ) + " x "
          + STRG( "[" ) + ITS( iOriginaVol->sizeZ ) + STRG( "]" ));

    /* @ Allocating cube array for the original volume */
   char*** originalCubeVol = Volume::allocCubeVolume_char
           (iOriginaVol->sizeX, iOriginaVol->sizeY, iOriginaVol->sizeZ);




    INFO("Packing the FLAT array in the CUBE ");

    /* @ Packing the original flat volume in the cube */
    Volume::packCubeVolume(originalCubeVol, iOriginaVol);

    /* @ Allocating the final volume */
    volume* iFinalSubVolume = (volume*) malloc (sizeof(volume));

    iFinalSubVolume->sizeX = iSubVolDim->max_X - iSubVolDim->min_X;
    iFinalSubVolume->sizeY = iSubVolDim->max_Y - iSubVolDim->min_Y;
    iFinalSubVolume->sizeZ = iSubVolDim->max_Z - iSubVolDim->min_Z;

    if (iFinalSubVolume->sizeX == iFinalSubVolume->sizeY
            && iFinalSubVolume->sizeX == iFinalSubVolume->sizeZ)
    {
        INFO("Final SUB-VOLUME has unified dimensions "
             + ITS(iFinalSubVolume->sizeX));

         iFinalSubVolume->sizeUni =  iFinalSubVolume->sizeZ;
    }
    else
    {
        INFO("Final SUB-VOLUME DOESN'T have unified dimensions "
             + ITS(iFinalSubVolume->sizeX));
        EXIT(0);
    }

    char*** finalCubeVol = Volume::allocCubeVolume_char
       (iFinalSubVolume->sizeX, iFinalSubVolume->sizeY, iFinalSubVolume->sizeZ);

    iFinalSubVolume->ptrVol_char =
            (char*) malloc (sizeof(char) * iFinalSubVolume->sizeX
                            * iFinalSubVolume->sizeY
                            * iFinalSubVolume->sizeZ);

    INFO("Allocating CUBE array for the final SUB-VOLUME: "
              + STRG( "[" ) + ITS( iFinalSubVolume->sizeX ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iFinalSubVolume->sizeY ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iFinalSubVolume->sizeZ ) + STRG( "]" ));



    INFO("Extractng the SUB-VOLUME");

    /* @ Extractig the SUB-VOLUME */
    Volume::extractSubVolume(originalCubeVol, finalCubeVol, iSubVolDim);

    //extractSubVolume_Flat(iOriginaVol->ptrVol_char, iFinalSubVolume->ptrVol_char, iSubVolDim);




    /* @ Dellocating the original cube volume */
   // FREE_MEM_3D_CHAR(originalCubeVol, iOriginaVol->sizeX, iOriginaVol->sizeY, iOriginaVol->sizeZ);

    /* @ Packing the final cube volume in the flat array */
    Volume::packFlatVolume(iFinalSubVolume, finalCubeVol);


    for (int y = 0; y < iOriginaVol->sizeY; y++)
    {
        for (int x = 0; x < iOriginaVol->sizeX; x++)
            free(originalCubeVol[y][x]);

        free(originalCubeVol[y]);
    }
    free(originalCubeVol);
    originalCubeVol = NULL;

    for (int y = 0; y < iFinalSubVolume->sizeY; y++)
    {
        for (int x = 0; x < iFinalSubVolume->sizeX; x++)
            free(finalCubeVol[y][x]);

        free(finalCubeVol[y]);
    }

    free(finalCubeVol);
    finalCubeVol = NULL;



    //FREE_MEM_3D_CHAR(finalCubeVol, iFinalSubVolume->sizeX, iFinalSubVolume->sizeY, iFinalSubVolume->sizeZ);

    INFO("Final volume extraction DONE");

    return iFinalSubVolume;
}

volume* Volume::createFloatVolume(volume* iVolume)
{
    INFO("Creating FLAT FLOAT32 volume : "
              + STRG( "[" ) + ITS( iVolume->sizeX ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolume->sizeY ) + STRG( "]" ) + " x "
              + STRG( "[" ) + ITS( iVolume->sizeZ ) + STRG( "]" ));

    /* @ Allocating float flat array */
    float* flatVol_float = (float*) malloc
            (sizeof(float*) * iVolume->sizeX * iVolume->sizeY * iVolume->sizeZ);

    /* @ Type conversion */
    for (int i = 0; i < iVolume->sizeX * iVolume->sizeY * iVolume->sizeZ; i++)
        flatVol_float[i] = (float) (unsigned char) iVolume->ptrVol_char[i];

    /* @ linking the float array to the original volume structure */
    iVolume->ptrVol_float = flatVol_float;

    INFO("Freeing the BYTE volume");

    /* @ freeing the BYTE volume */
    free((iVolume->ptrVol_char));

    INFO("Creating FLAT FLOAT32 volume DONE");

    return iVolume;
}
