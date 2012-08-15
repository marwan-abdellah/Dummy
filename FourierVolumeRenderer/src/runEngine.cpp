#include "shared.h"
#include "OpenGL/cOpenGL.h"
#include "SpectrumProcessing/Spectrum.h"
#include "Loader/Loader.h"
#include "VolumeProcessing/volume.h"
#include "FFTShift/FFTShift.h"
#include "WrappingAround/WrappingAround.h"
#include "OpenGL/DisplayList.h"
#include "SliceProcessing/Slice.h"
#include "FFTShift/FFTShift.h"
#include "RenderingLoop/RenderingLoop.h"
#include "eFourierVolRen.h"

char* eVolPath = "/home/abdellah/Software/DataSets/CTData/CTData";

int main(int argc, char** argv)
{ 	
    /*@ Run the rendering engine */
    eFourierVolRen::run(argc, argv, eVolPath);

	return 0;
}


