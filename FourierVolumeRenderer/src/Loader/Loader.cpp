#include "Loader.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

volDim* Loader::loadHeader(const char *path)
{
    LOG();

    /* @ Allocation structure */
    volDim* iVolDim = (volDim*) malloc (sizeof(volDim));

    /* @ Getting the header file "<FILE_NAME>.hdr" */
    char hdrFile[1000];
    sprintf(hdrFile, "%s.hdr", path);

    /* @ Input file stream */
    std::ifstream ifile;

    /* @ Open file */
    ifile.open(hdrFile, std::ios::in);

    /* @ Double checking for the existance of the header file */
    if (!ifile.fail())
    {
        /* @ Check point success */
        INFO("Openging header file : " + CCATS(hdrFile));
    }
    else
    {
        /* @ Check point failing */
        INFO("Error OPENING header file : " + CCATS( hdrFile));
        EXIT( 0 );
    }

    /* @ Reading the dimensions in XYZ order */
    ifile >> iVolDim->size_X;
    ifile >> iVolDim->size_Y;
    ifile >> iVolDim->size_Z;

    INFO("Volume Dimensions : "
        + STRG( "[" ) + ITS( iVolDim->size_X ) + STRG( "]" ) + " x "
        + STRG( "[" ) + ITS( iVolDim->size_Y ) + STRG( "]" ) + " x "
        + STRG( "[" ) + ITS( iVolDim->size_Z ) + STRG( "]" ));

    /* @ Closing the innput stream */
    ifile.close();

    INFO("Reading volume header DONE");

    return iVolDim;
}

volume* Loader::loadVolume(const char* path)
{
    LOG();

    /* @ Loading the volume file */
    char volFile[1000];
    sprintf(volFile, "%s.img", path);

    /* @ Checking for the existance of the volume file */
    if (!volFile)
    {
        /* @ Check point failing */
        INFO("Error FINDING raw volume file : " + CCATS(volFile));
        EXIT( 0 );
    }
    else
    {
        /* @ Check point success */
        INFO("Opening raw volume file : " + CCATS(volFile));
    }

    /* @ Allocating volume structre */
    volume* iVolume = (volume*) malloc (sizeof(volume));

    /* @ Reading the header file to get volume dimensions */
    volDim* iVolDim = loadHeader(path);
    iVolume->sizeX = iVolDim->size_X;
    iVolume->sizeY = iVolDim->size_Y;
    iVolume->sizeZ = iVolDim->size_Z;

    if (iVolume->sizeX == iVolume->sizeY && iVolume->sizeX == iVolume->sizeZ)
    {
        iVolume->sizeUni = iVolume->sizeX;
        INFO("Loaded volume has unified dimensiosn of:" + ITS(iVolume->sizeUni));
    }
    else
    {
        INFO("NON UNIFIED VOLUME HAS BEEN LOADDED - UNIFICATION REQUIRED");
    }

    /* @ Volume flat size */
    iVolume->volSize = iVolume->sizeX * iVolume->sizeY * iVolume->sizeZ;
    INFO("Volume flat size : " + ITS(iVolume->volSize));

    /* @Volume size in bytes */
    iVolume->volSizeBytes = sizeof(char) * iVolume->volSize;
    INFO("Volume size in MBbytes: " + FTS(iVolume->volSize / (1024 * 1024)));

    /* @ Allocating volume */
    iVolume->ptrVol_char = (char*) malloc (iVolume->volSizeBytes);
    INFO("Preparing volume data & meta-data");

    /* @ Opening volume file */
    FILE* ptrFile = fopen(volFile, "rb");

     // Double checking for the existance of the volume file
    if (!ptrFile)
    {
        /* @ Check point failing */
        INFO("Error FINDING raw volume file : " + CCATS(volFile));
        EXIT( 0 );
    }

    // Read the volume raw file
    size_t imageSize = fread(iVolume->ptrVol_char,
                             BYTE,
                             iVolume->volSizeBytes,
                             ptrFile);

    // Checking if the volume was loaded or not
    if (!imageSize)
    {
        INFO("Error READING raw volume file : " + CCATS(volFile));
        EXIT(0);
    }

    INFO("Reading volume raw file DONE");

    return iVolume;
}
