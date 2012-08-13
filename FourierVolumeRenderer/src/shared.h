#ifndef SHARED_H
#define SHARED_H

#define BYTE 1

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>

struct subVolDim
{
    int min_X;
    int max_X;
    int min_Y;
    int max_Y;
    int min_Z;
    int max_Z;
};

struct volDim
{
    int size_X;
    int size_Y;
    int size_Z;
};

struct volume
{
    char* ptrVol_char;
    float* ptrVol_float;
    double* ptrVol_double;

    int sizeX;
    int sizeY;
    int sizeZ;
    int sizeUni;

    int volSize;
    int volSizeBytes;
};

#endif // SHARED_H
