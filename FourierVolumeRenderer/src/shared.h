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

/* @ ImageMagick includes */
#include <ImageMagick/Magick++.h>
#include <ImageMagick/Magick++/Blob.h>
#include <ImageMagick/Magick++/CoderInfo.h>
#include <ImageMagick/Magick++/Color.h>
#include <ImageMagick/Magick++/Drawable.h>
#include <ImageMagick/Magick++/Exception.h>
#include <ImageMagick/Magick++/Geometry.h>
#include <ImageMagick/Magick++/Image.h>
#include <ImageMagick/Magick++/Include.h>
#include <ImageMagick/Magick++/Montage.h>
#include <ImageMagick/Magick++/Pixels.h>
#include <ImageMagick/Magick++/STL.h>
#include <ImageMagick/Magick++/TypeMetric.h>

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

struct sliceDim
{
    int size_X;
    int size_Y;
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

struct compositionImages
{
    Magick::Image* image_H1;
    Magick::Image* image_H2;
    Magick::Image* image_Final;
};

#endif // SHARED_H
