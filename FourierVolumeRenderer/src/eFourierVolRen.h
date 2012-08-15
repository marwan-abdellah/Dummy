#ifndef E_FOURIER_VOL_REN_H
#define E_FOURIER_VOL_REN_H

#include "shared.h"

using namespace Magick;

namespace eFourierVolRen
{
    void run(int argc, char** argv, char* iVolPath);
    volume* loadingVol(char* iVolPath);
    void initContexts(int argc, char** argv);
    sliceDim* preProcess(volume* iVolume_char);
    Magick::Image* rendering(const int iSliceWidth, const int iSliceHeight);

    /* @ */
    volume* decomposeVolume(volume* iVolume, int iBrickIndex);

    /* @ */
    compositionImages* createFinalImage(int iFinalSliceWidth, int iFinalSliceHeight);

    void addTileToFinalImage(int iFinalSliceWidth,
                                             int iFinalSliceHeight,
                                             Image* iTile,
                                             compositionImages* iImageList,
                                             int ibrickIndex);

    void setBrick(Image* iTile, compositionImages* iImageList,
                  int iXOffset, int iYOffset,
                  int iTileWidth, int iTileHeight,
                  int iH);

    void writeFinalImageToDisk(compositionImages* iImageList);
}

#endif // E_FOURIER_VOL_REN_H
