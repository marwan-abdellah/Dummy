#ifndef VOLUME_H
#define VOLUME_H

#include "shared.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

/* @ Volume namespace */
namespace Volume
{
    /* @ */
    char*** allocCubeVolume_char(const int size_X,const int size_Y,const int size_Z);

    /* @ */
    void packFlatVolume(volume* iVolume, char*** cubeVolume);

    /* @ */
    void packCubeVolume(char*** cubeVolume, volume* iVolume);

    /* @ */
    void extractSubVolume(char*** originalCubeVol, char*** finalCubeVol, const subVolDim* iSubVolDim);

    /* @ */
    volume* extractFinalVolume(volume* originalFlatVol, const subVolDim* iSubVol);

    /* @ */
    volume* createFloatVolume(volume* iVolume_char);

    /* @ */
    void unifyVolumeDim(volume* iVolume);

    /* @ */
    void extractSubFlatVolume(volume* originalFlatVol,
                                  volume* finalFlatVol,
                                  const subVolDim* iSubVolDim);
    /* @ */
    int getUnifiedDimension(int iMaxDim);

    int prepareVolumeArrays(volume* iOriginaVol, int iFinalDim);

}

#endif // VOLUME_H
