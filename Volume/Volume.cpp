/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "Volume.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"


void Volume::testVolume()
{

}

volumeDimensions_t Volume::openVolHeader(const char* volName, const char* volPath)
{
	LOG();

	// Forming header file name with extension ".hdr"
	char volHdrFile [1000];
	strcpy(volHdrFile, volPath);
	strcat(volHdrFile, "/");
	strcat(volHdrFile, volName);
	strcat(volHdrFile, ".hdr");

	// Allocate volume dimensions structure for retrieving volume dimensions
	// from the header file
	volumeDimensions_t volDim = MEM_ALLOC_1D_GENERIC(volumeDimensions, 1);

	// Input stream for reading the header file
	istream_t volHdrFileStream;

	// Open the header file
	volHdrFileStream.open(volHdrFile, ios::in);

    // If failure, exit
	if (volHdrFileStream.fail())
	{
        INFO( "Could not open " + CATS( volHdrFile ));
        EXIT( 0 );
	}
    else {
    	INFO("Opening volume header file : " + STRG( "[" ) + CATS(volHdrFile) + STRG( "]" ));
    }

	// Retrieve volume dimensions form the file into the structure
    volHdrFileStream >> (volDim->size_X);
    volHdrFileStream >> (volDim->size_Y);
    volHdrFileStream >> (volDim->size_Z);

    INFO("Volume Dimensions : "
         + STRG( "[" ) + ITS( volDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( volDim->size_X ) + STRG( "]" ) + " x "
         + STRG( "[" ) + ITS( volDim->size_X ) + STRG( "]" ));

	// Close the header file stream
	volHdrFileStream.close();
	// info about the closure

	return volDim;
}

vol_char_t Volume::openVolumeFile(const char* volName, const char* volPath, volumeDimensions_t volDim)
{
	LOG();

	// Forming volume file name with extension ".img"
	char volFile [1000];
	strcpy(volFile, volPath);
	strcat(volFile, "/");
	strcat(volFile, volName);
	strcat(volFile, ".img");

	// Open the volume file with "read binary" mode
	// Pointer to the volume file
    FILE* ptrVol = fopen(volFile, "rb");

    // Check for a valid file
    if (!ptrVol) {
        INFO("Error opening volume " + CCATS( volFile));
        EXIT( 0 );
    }
    else {

    	// Volume size (X * Y * Z)
    	const int volSize = (volDim->size_X * volDim->size_Y * volDim->size_Z);

    	// Allocating volume image "char" array
    	vol_char_t volImg = MEM_ALLOC_1D_GENERIC(vol_char, volSize);

    	// Reading the volume into the array
    	const size_t fileSize_Bytes = fread(volImg, 1, volSize, ptrVol);
    	return volImg ;
    }
    return NULL;
}


volume_char_t Volume::loadVolume(const char* volName, const char* volPath)
{
	LOG();

	// Volume dimensions
	volumeDimensions_t volDim = MEM_ALLOC_1D_GENERIC(volumeDimensions, 1);

	// Opening header file to retrieve volume dimensions
	volDim = openVolHeader(volName, volPath);

	// Allocating volume structure
	volume_char_t volData = MEM_ALLOC_1D_GENERIC(volume_char, 1);

	// Loading volume data
	volData->volDim = volDim;
	volData->volImg = openVolumeFile(volName, volPath, volDim);

	return volData;
}

void Volume::createVolume_float(const int volSize, char* charVol)
{
	LOG();

	/* Allocate float array for the input volume to increase precision */
	float* floatVol = (float*) malloc ( volSize * sizeof(float));

	/* Packing the extracted volume in the float 1D array */
	printf("Packing Data in a Float Array ... \n");

	for (int i = 0; i < volSize; ++i)
	{
		floatVol[i] = (float) (unsigned char) charVol[i];
	}
	printf("	Packing is Done  \n");

	/* Realeasing volume in char array */
	printf("	Releasing Byte Data 	\n\n");
	//delete [] mVolumeData;
}

int Volume::freeVolume_1D_char(char* charVol)
{
	delete [] charVol;
	return 0;
}

int Volume::freeVolume_1D_float(float* floatVol)
{
	delete [] floatVol;
	return 0;
}

int Volume::freeVolume_3D_char(char*** charVol)
{
	return 0;
}

int Volume::freeVolume_3D_float(float*** floatVol)
{
	return 0;
}


char*** Volume::allocateVolume_3D_char(int volSize_X, int volSize_Y, int volSize_Z)
{
	LOG();
	return MEM_ALLOC_3D_CHAR(volSize_X, volSize_Y, volSize_Z);
}

float*** Volume::allocateVolume_3D_float(int volSize_X, int volSize_Y, int volSize_Z)
{
	LOG();
	return MEM_ALLOC_3D_FLOAT(volSize_X, volSize_Y, volSize_Z);
}

char*** Volume::convertVolume_1D_To_3D_char(const char* inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z)
{
	char*** outputVol;
	return outputVol;
}

float*** Volume::convertVolume_1D_To_3D_float(const float* inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z)
{
	float*** outputVol;
	return outputVol;
}

char* Volume::convertVolume_3D_To_1D_char(const char*** inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z)
{
	char* outputVol;
	return outputVol;
}

float* Volume::convertVolume_3D_To_1D_float(const float*** inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z)
{
	float* outputVol;
	return outputVol;
}

char*** Volume::extractVolume_char(const char*** inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z,
		const int start_X, const int start_Y, const int start_Z)
{
	char*** outputVol;


	for (int i = 0; i < volSize_X; ++i) {
		for (int j = 0; j < volSize_Y; ++j) {
			for (int k = 0; k < volSize_Z; ++k) {

				outputVol[i][j][k] = inputVol[(start_X) + i][(start_Y) + j][(start_Z) + k];
			}
		}
	}

	return outputVol;
}

float*** Volume::extractVolume_float(const float*** inputVol,
		const int volSize_X, const int volSize_Y, const int volSize_Z,
		const int start_X, const int start_Y, const int start_Z)
{
	float*** outputVol;

	return outputVol;
}


volume_complex_float_t
Volume::createComplexVolume_float(volume_float_t volReal)
{
	LOG();

	const int volSize_XYZ = (volReal->volDim->size_X *
							 volReal->volDim->size_X *
							 volReal->volDim->size_X);

	//
	volume_complex_float_t volComplex = MEM_ALLOC_1D_GENERIC(volume_complex_float, 1);
	volComplex->volImg = MEM_ALLOC_1D_GENERIC(fftwf_complex, volSize_XYZ);

	for (int i = 0; i < volSize_XYZ; ++i)
	{
		volComplex->volImg[i][0] = volReal->volImg[0];
		volComplex->volImg[i][1] = 0;

	}
	return volComplex;
}


volume_complex_double_t
Volume::createComplexVolume_double(volume_double_t volReal)
{
	LOG();

	const int volSize_XYZ = (volReal->volDim->size_X *
							 volReal->volDim->size_X *
							 volReal->volDim->size_X);

	//
	volume_complex_double_t volComplex = MEM_ALLOC_1D_GENERIC(volume_complex_double, 1);
	volComplex->volImg = MEM_ALLOC_1D_GENERIC(fftw_complex, volSize_XYZ);

	for (int i = 0; i < volSize_XYZ; ++i)
	{
		volComplex->volImg[i][0] = volReal->volImg[0];
		volComplex->volImg[i][1] = 0;

	}
	return volComplex;
}

volume_complex_float_t
Volume::createComplexVolumeFromChar_float(volume_char_t volReal)
{
	LOG();

	const int volSize_XYZ = (volReal->volDim->size_X *
							 volReal->volDim->size_X *
							 volReal->volDim->size_X);

	//
	volume_complex_float_t volComplex = MEM_ALLOC_1D_GENERIC(volume_complex_float, 1);
	volComplex->volImg = MEM_ALLOC_1D_GENERIC(fftwf_complex, volSize_XYZ);
	volComplex->volDim = MEM_ALLOC_1D_GENERIC(volumeDimensions, 1);

	for (int i = 0; i < volSize_XYZ; ++i) {
		volComplex->volImg[i][0] = static_cast<float> (volReal->volImg[0]);
		volComplex->volImg[i][1] = 0;
	}

	*(volComplex->volDim) = *(volReal->volDim);

	return volComplex;
}

volume_complex_double_t
Volume::createComplexVolumeFromChar_double(volume_char_t volReal)
{
	LOG();

	const int volSize_XYZ = (volReal->volDim->size_X *
							 volReal->volDim->size_X *
							 volReal->volDim->size_X);

	//
	volume_complex_double_t volComplex = MEM_ALLOC_1D_GENERIC(volume_complex_double, 1);
	volComplex->volImg = MEM_ALLOC_1D_GENERIC(fftw_complex, volSize_XYZ);

	for (int i = 0; i < volSize_XYZ; ++i)
	{
		volComplex->volImg[i][0] = static_cast<double> (volReal->volImg[0]);
		volComplex->volImg[i][1] = 0;

	}
	return volComplex;
}
