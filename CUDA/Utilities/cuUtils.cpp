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
#include "Globals.h"

int cuUtils::upload_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL" + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(float);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_float(float* hostArr, float* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_float(float* hostArr, float* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(float);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_double(double* hostArr, double* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_double(double* hostArr, double* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(double);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_double(double* hostArr, double* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_double(double* hostArr, double* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(double);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}


	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::upload_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::upload_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);

	if(devMem < 1024)
	{
		INFO("Memory Required: " + ITS(devMem) + " Bytes");
	}
	else
	{
		if (devMem < 1024 * 1024)
		{
			INFO("Memory Required: " + ITS(devMem/ 1024) + " KBytes");
		}
		else
		{
			if (devMem < 1024 * 1024 * 1204)
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024)) + " MBytes");
			}
			else
			{
				INFO("Memory Required: " + ITS(devMem/ (1024 * 1024 * 1024)) + " GBytes");
			}
		}
	}

	if (2 * devMem >= MAX_GPU_MEMORY)
	{
		INFO("MEMORY WALL: " + ITS(MAX_GPU_MEMORY));
	}

	cutilSafeCall(cudaMemcpy(devArr, hostArr, devMem, cudaMemcpyHostToDevice));

	return 0;
}

int cuUtils::download_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X)
{
	LOG();

	int devMem = size_X * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y)
{
	LOG();

	int devMem = size_X * size_Y * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}

int cuUtils::download_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z)
{
	LOG();

	int devMem = size_X * size_Y * size_Z * sizeof(cufftDoubleComplex);

	cutilSafeCall(cudaMemcpy(hostArr, devArr, devMem, cudaMemcpyDeviceToHost));

	return 0;
}
