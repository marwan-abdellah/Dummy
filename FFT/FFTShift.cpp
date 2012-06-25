/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: FFTShift.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "FFTShift.h"
#include "Timers/BoostTimers.h"

float* FFT_Shift_1D_float(float* input, int nX, durationStruct* duration)
{
	LOG();

	const int N = nX;

	/* Timing parameters */
	time_boost start, end;
	durationStruct* resDuration;

	// Start timer
	start = Timers::BoostTimers::getTime_MicroSecond();

	float* output;
	output = MEM_ALLOC_1D (float, N);

	for(int i = 0; i < N/2; i++)
	{
		output[(N/2) + i] = input[i];
		output[i] = input[(N/2) + i];
	}

	// Start timer
	end = Timers::BoostTimers::getTime_MicroSecond();

	// Calculate the duration
	resDuration = Timers::BoostTimers::getDuration(start, end);

	*duration = *resDuration;

	return output;
}

float** FFT::FFT_Shift_2D_float(float** input, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		float** output;
		output = MEM_ALLOC_2D_FLOAT(N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				output[(N/2) + i][(N/2) + j] = input[i][j];
				output[i][j] = input[(N/2) + i][(N/2) + j];

				output[i][(N/2) + j] = input[(N/2) + i][j];
				output[(N/2) + i][j] = input[i][(N/2) + j];
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}

	return NULL;
}

float*** FFT::FFT_Shift_3D_float(float*** input, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		float ***output;;
		output = MEM_ALLOC_3D_FLOAT(N, N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		/* Doing the 3D FFT shift operation */
		for (int k = 0; k < N/2; k++)
			for (int i = 0; i < N/2; i++)
				for(int j = 0; j < N/2; j++)
				{
					output[(N/2) + i][(N/2) + j][(N/2) + k] = input[i][j][k];
					output[i][j][k] = input[(N/2) + i][(N/2) + j][(N/2) + k];

					output[(N/2) + i][j][(N/2) + k] = input[i][(N/2) + j][k];
					output[i][(N/2) + j][k] = input[(N/2) + i][j][(N/2) + k];

					output[i][(N/2) + j][(N/2) + k] = input[(N/2) + i][j][k];
					output[(N/2) + i][j][k] = input[i][(N/2) + j][(N/2) + k];

					output[i][j][(N/2) + k] = input[(N/2) + i][(N/2) + j][k];
					output[(N/2) + i][(N/2) + j][k] = input[i][j][(N/2) + k];
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

float* FFT::repack_2D_float(float** input_2D, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		float *output_1D;;
		output_1D = MEM_ALLOC_1D_FLOAT(N  * N);

		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				output_1D[ctr] = input_2D[i][j];
				ctr++;
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;


		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}
	return NULL;
}

float* FFT::repack_3D_float(float*** input_3D, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		float *output_1D;;
		output_1D = MEM_ALLOC_1D_FLOAT(N * N * N);

		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		// Re-packing the 3D volume into 1D array
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
				{
					output_1D[ctr] = input_3D[i][j][k];
					ctr++;
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

double* FFT_Shift_1D_double(double* input, int nX, durationStruct* duration)
{
	LOG();

	const int N = nX;

	double* output;
	output = MEM_ALLOC_1D(double, N);

	/* Timing parameters */
	time_boost start, end;
	durationStruct* resDuration;

	// Start timer
	start = Timers::BoostTimers::getTime_MicroSecond();

	for(int i = 0; i < N/2; i++)
	{
		output[(N/2) + i] = input[i];
		output[i] = input[(N/2) + i];
	}

	// Start timer
	end = Timers::BoostTimers::getTime_MicroSecond();

	// Calculate the duration
	resDuration = Timers::BoostTimers::getDuration(start, end);

	*duration = *resDuration;

	return output;
}

double** FFT::FFT_Shift_2D_double(double** input, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		double** output;
		output = MEM_ALLOC_2D_DOUBLE(N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				output[(N/2) + i][(N/2) + j] = input[i][j];
				output[i][j] = input[(N/2) + i][(N/2) + j];

				output[i][(N/2) + j] = input[(N/2) + i][j];
				output[(N/2) + i][j] = input[i][(N/2) + j];
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}

	return NULL;
}

double*** FFT::FFT_Shift_3D_double(double*** input, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		double ***output;;
		output = MEM_ALLOC_3D_DOUBLE(N, N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		/* Doing the 3D FFT shift operation */
		for (int k = 0; k < N/2; k++)
			for (int i = 0; i < N/2; i++)
				for(int j = 0; j < N/2; j++)
				{
					output[(N/2) + i][(N/2) + j][(N/2) + k] = input[i][j][k];
					output[i][j][k] = input[(N/2) + i][(N/2) + j][(N/2) + k];

					output[(N/2) + i][j][(N/2) + k] = input[i][(N/2) + j][k];
					output[i][(N/2) + j][k] = input[(N/2) + i][j][(N/2) + k];

					output[i][(N/2) + j][(N/2) + k] = input[(N/2) + i][j][k];
					output[(N/2) + i][j][k] = input[i][(N/2) + j][(N/2) + k];

					output[i][j][(N/2) + k] = input[(N/2) + i][(N/2) + j][k];
					output[(N/2) + i][(N/2) + j][k] = input[i][j][(N/2) + k];
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;


		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

double* FFT::repack_2D_double(double** input_2D, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		double *output_1D;;
		output_1D = MEM_ALLOC_1D(double, N  * N);

		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				output_1D[ctr] = input_2D[i][j];
				ctr++;
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}
	return NULL;
}

double* FFT::repack_3D_double(double*** input_3D, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		double *output_1D;;
		output_1D = MEM_ALLOC_1D(double, N * N * N);


		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		// Re-packing the 3D volume into 1D array
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
				{
					output_1D[ctr] = input_3D[i][j][k];
					ctr++;
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}


cuComplex* FFT_Shift_1D_cuComplex(cuComplex* input, int nX, durationStruct* duration)
{
	LOG();

	const int N = nX;

	cuComplex* output;
	output = MEM_ALLOC_1D(cuComplex, N);

	/* Timing parameters */
	time_boost start, end;
	durationStruct* resDuration;

	// Start timer
	start = Timers::BoostTimers::getTime_MicroSecond();

	for(int i = 0; i < N/2; i++)
	{
		output[(N/2) + i] = input[i];
		output[i] = input[(N/2) + i];
	}

	// Start timer
	end = Timers::BoostTimers::getTime_MicroSecond();

	// Calculate the duration
	resDuration = Timers::BoostTimers::getDuration(start, end);

	*duration = *resDuration;

	return output;
}

cuComplex** FFT::FFT_Shift_2D_cuComplex(cuComplex** input, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		cuComplex** output;
		output = MEM_ALLOC_2D_CUFFTCOMPLEX(N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				output[(N/2) + i][(N/2) + j] = input[i][j];
				output[i][j] = input[(N/2) + i][(N/2) + j];

				output[i][(N/2) + j] = input[(N/2) + i][j];
				output[(N/2) + i][j] = input[i][(N/2) + j];
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}

	return NULL;
}

cuComplex*** FFT::FFT_Shift_3D_cuComplex(cuComplex*** input, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		cuComplex ***output;;
		output = MEM_ALLOC_3D_CUFFTCOMPLEX(N, N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		/* Doing the 3D FFT shift operation */
		for (int k = 0; k < N/2; k++)
			for (int i = 0; i < N/2; i++)
				for(int j = 0; j < N/2; j++)
				{
					output[(N/2) + i][(N/2) + j][(N/2) + k] = input[i][j][k];
					output[i][j][k] = input[(N/2) + i][(N/2) + j][(N/2) + k];

					output[(N/2) + i][j][(N/2) + k] = input[i][(N/2) + j][k];
					output[i][(N/2) + j][k] = input[(N/2) + i][j][(N/2) + k];

					output[i][(N/2) + j][(N/2) + k] = input[(N/2) + i][j][k];
					output[(N/2) + i][j][k] = input[i][(N/2) + j][(N/2) + k];

					output[i][j][(N/2) + k] = input[(N/2) + i][(N/2) + j][k];
					output[(N/2) + i][(N/2) + j][k] = input[i][j][(N/2) + k];
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

cuComplex* FFT::repack_2D_cuComplex(cuComplex** input_2D, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		cuComplex *output_1D;;
		output_1D = MEM_ALLOC_1D(cuComplex, N  * N);

		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				output_1D[ctr] = input_2D[i][j];
				ctr++;
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}
	return NULL;
}

cuComplex* FFT::repack_3D_cuComplex(cuComplex*** input_3D, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		cuComplex *output_1D;;
		output_1D = MEM_ALLOC_1D(cuComplex, N * N * N);


		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		// Re-packing the 3D volume into 1D array
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
				{
					output_1D[ctr] = input_3D[i][j][k];
					ctr++;
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

cuDoubleComplex* FFT_Shift_1D_cuDoubleComplex(cuDoubleComplex* input, int nX, durationStruct* duration)
{
	LOG();

	const int N = nX;

	cuDoubleComplex* output;
	output = MEM_ALLOC_1D(cuDoubleComplex, N);

	/* Timing parameters */
	time_boost start, end;
	durationStruct* resDuration;

	// Start timer
	start = Timers::BoostTimers::getTime_MicroSecond();

	for(int i = 0; i < N/2; i++)
	{
		output[(N/2) + i] = input[i];
		output[i] = input[(N/2) + i];
	}

	// Start timer
	end = Timers::BoostTimers::getTime_MicroSecond();

	// Calculate the duration
	resDuration = Timers::BoostTimers::getDuration(start, end);

	*duration = *resDuration;

	return output;
}

cuDoubleComplex** FFT::FFT_Shift_2D_cuDoubleComplex(cuDoubleComplex** input, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		cuDoubleComplex** output;
		output = MEM_ALLOC_2D_CUFFTDOUBLECOMPLEX(N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				output[(N/2) + i][(N/2) + j] = input[i][j];
				output[i][j] = input[(N/2) + i][(N/2) + j];

				output[i][(N/2) + j] = input[(N/2) + i][j];
				output[(N/2) + i][j] = input[i][(N/2) + j];
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}

	return NULL;
}

cuDoubleComplex*** FFT::FFT_Shift_3D_cuDoubleComplex(cuDoubleComplex*** input, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		cuDoubleComplex ***output;;
		output = MEM_ALLOC_3D_CUFFTDOUBLECOMPLEX(N, N, N);

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		/* Doing the 3D FFT shift operation */
		for (int k = 0; k < N/2; k++)
			for (int i = 0; i < N/2; i++)
				for(int j = 0; j < N/2; j++)
				{
					output[(N/2) + i][(N/2) + j][(N/2) + k] = input[i][j][k];
					output[i][j][k] = input[(N/2) + i][(N/2) + j][(N/2) + k];

					output[(N/2) + i][j][(N/2) + k] = input[i][(N/2) + j][k];
					output[i][(N/2) + j][k] = input[(N/2) + i][j][(N/2) + k];

					output[i][(N/2) + j][(N/2) + k] = input[(N/2) + i][j][k];
					output[(N/2) + i][j][k] = input[i][(N/2) + j][(N/2) + k];

					output[i][j][(N/2) + k] = input[(N/2) + i][(N/2) + j][k];
					output[(N/2) + i][(N/2) + j][k] = input[i][j][(N/2) + k];
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}

cuDoubleComplex* FFT::repack_2D_cuDoubleComplex(cuDoubleComplex** input_2D, int nX, int nY, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY)
	{
		const int N = nX;

		cuDoubleComplex *output_1D;;
		output_1D = MEM_ALLOC_1D(cuDoubleComplex, N  * N);

		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				output_1D[ctr] = input_2D[i][j];
				ctr++;
			}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 2D FFT shift yet");
		EXIT(0);
	}
	return NULL;
}

cuDoubleComplex* FFT::repack_3D_cuDoubleComplex(cuDoubleComplex*** input_3D, int nX, int nY, int nZ, durationStruct* duration)
{
	LOG();

	// Only supporting a unified FFT shift for the time being
	if (nX == nY & nX == nZ)
	{
		const int N = nX;

		cuDoubleComplex *output_1D;;
		output_1D = MEM_ALLOC_1D(cuDoubleComplex, N * N * N);


		int ctr = 0;

		/* Timing parameters */
		time_boost start, end;
		durationStruct* resDuration;

		// Start timer
		start = Timers::BoostTimers::getTime_MicroSecond();

		// Re-packing the 3D volume into 1D array
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
				{
					output_1D[ctr] = input_3D[i][j][k];
					ctr++;
				}

		// Start timer
		end = Timers::BoostTimers::getTime_MicroSecond();

		// Calculate the duration
		resDuration = Timers::BoostTimers::getDuration(start, end);

		*duration = *resDuration;

		return output_1D;
	}
	else
	{
		INFO("We do NOT support non-unified 3D FFT shift yet");
		EXIT(0);
	}
	return NULL;

}
