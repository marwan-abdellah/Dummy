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

char* eVolPath = "/home/abdellah/Software/DataSets/VHM_SHOULDER_8/VHM_SHOULDER_8";

using namespace std;

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"


int main(int argc, char** argv)
{ 	
    /*@ Run the rendering engine */
    eFourierVolRen::run(argc, argv, eVolPath);

	return 0;
}


/*
    // Dicom format
    DcmFileFormat fileformat;

    // Load file
    OFCondition status = fileformat.loadFile("TestCase.dcm");

    // Check if the file is loaded correctly
    if (status.good())
    {
        // patient name
        OFString patientsName;

        if (fileformat.getDataset()->findAndGetOFString(DCM_PatientsName, patientsName).good())
        {
            cout << "Patient's Name: " << patientsName << endl;
        }
        else
        {
            cerr << "Error: cannot access Patient's Name!" << endl;
        }
    }
    else
    {
        cerr << "Error: cannot read DICOM file (" << status.text() << ")" << endl;
    }
*/
