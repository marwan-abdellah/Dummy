/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : Volume
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/
#include "cuUtils.h"

int cuUtils::upload_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

	const int devMem = size_X * sizeof(float);

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	const long devMem = size_X * size_Y * sizeof(float);

	COUT << "Memory: " << ((devMem) / (1024 * 1024)) << " MBytes" << ENDL;

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	const int devMem = size_X * size_Y * size_Z * sizeof(float);

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

	const int devMem = size_X * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	const int devMem = size_X * size_Y * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	const int devMem = size_X * size_Y * size_Z * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}
