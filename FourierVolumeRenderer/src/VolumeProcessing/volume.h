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
    char*** allocCubeVolume_char(const int size_X,const int size_Y,const int size_Z);
    float*** allocCubeVolume_float(const int size_X, const int size_Y, const int size_Z);
    void packFlatVolume(char* flatVolume, char*** cubeVolume, const volDim* iVolDim);
    void packCubeVolume(char*** cubeVolume, char* flatVolume, const volDim* iVolDim);
    void extractSubVolume(char*** originalCubeVol, char*** finalCubeVol, const subVolDim* iSubVolDim);
    char* extractFinalVolume(char* originalFlatVol, const volDim* originalDim, const subVolDim* iSubVol);
    float* createFloatVolume(char* flatVol_char, const volDim* iVolDim);
}

#endif // VOLUME_H
