#ifndef LOADER_H
#define LOADER_H

#include "shared.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

/*@ Loader namespace */
namespace Loader
{
    /* @ Loading volume header file */
    volDim* loadHeader(const char* pathToHeaderFile);

    /* @ Loading volume image file */
    volume* loadVolume(const char* pathToVolumeFile);
}

#endif // LOADER_H
