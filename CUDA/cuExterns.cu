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
 
#ifndef CU_EXTERNS_CU_
#define CU_EXTERNS_CU_

#include "cuGlobals.h"

#include "cuCopyArray.cu"
#include "FFT/Real/cuFFTShift_2D_Real.cu"
#include "FFT/Real/cuFFTShift_3D_Real.cu"
#include "FFT/Complex/cuFFTShift_2D_Complex.cu"
#include "FFT/Complex/cuFFTShift_3D_Complex.cu"
#include "Timers/BoostTimers.h"


extern 
void cuCopyArray(dim3 cuBlock, dim3 cuGrid, float* devArrayOutput, float* devArrayInput, int nX)
{
    copyArray_2D_float_kernel <<< cuGrid, cuBlock>>> ( devArrayOutput, devArrayInput,  nX); 
} 

extern 
void cuFFTShift_2D( dim3 cuBlock, dim3 cuGrid,
                    float* devArrayOutput, float* devArrayInput, 
                    int nX, 
                    cudaProfile* cuProfile)
{   
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
     
    fftShift_2D_Kernel <<< cuGrid, cuBlock >>> (devArrayOutput, devArrayInput, nX);
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
}

extern 
void cuFFTShift_2D_Double( dim3 cuBlock, dim3 cuGrid, 
                           double* devArrayOutput, double* devArrayInput, 
                           int nX, 
                           cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_2D_Double_Kernel <<< cuGrid, cuBlock>>> (devArrayOutput, devArrayInput, nX); 
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
}

extern 
void cuFFTShift_2D_Complex( dim3 cuBlock, dim3 cuGrid, 
                            cufftComplex* devArrayOutput, cufftComplex* devArrayInput, 
                            int nX,
                            cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_2D_Complex_Kernel <<< cuGrid, cuBlock>>> (devArrayOutput, devArrayInput, nX);
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError(); 
}

extern 
void cuFFTShift_2D_Double_Complex( dim3 cuBlock, dim3 cuGrid, 
                                   cufftDoubleComplex* devArrayOutput, cufftDoubleComplex* devArrayInput, 
                                   int nX,
                                   cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_2D_Double_Complex_Kernel <<< cuGrid, cuBlock>>> (devArrayOutput, devArrayInput, nX); 
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError(); 
}

extern 
void cuFFTShift_3D( dim3 cuBlock, dim3 cuGrid, 
                    float* devArrayOutput, float* devArrayInput, 
                    int nX, 
                    cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_3D_i(devArrayInput, devArrayOutput, nX, cuBlock, cuGrid); 
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError(); 
}

extern 
void cuFFTShift_3D_Double( dim3 cuBlock, dim3 cuGrid, 
                           double* devArrayOutput, double* devArrayInput, 
                           int nX, 
                           cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_3D_Double_i(devArrayInput, devArrayOutput, nX, cuBlock, cuGrid);
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();  
}

extern 
void cuFFTShift_3D_Complex( dim3 cuBlock, dim3 cuGrid, cufftComplex* devArrayOutput, cufftComplex* devArrayInput, 
                            int nX, 
                            cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_3D_Complex_i(devArrayInput, devArrayOutput, nX, cuBlock, cuGrid); 
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError(); 
}

extern 
void cuFFTShift_3D_Double_Complex( dim3 cuBlock, dim3 cuGrid, cufftDoubleComplex* devArrayOutput, cufftDoubleComplex* devArrayInput, 
                                   int nX, 
                                   cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    fftShift_3D_Double_Complex_i(devArrayInput, devArrayOutput, nX, cuBlock, cuGrid); 
    cudaThreadSynchronize(); 

    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError(); 
}

#endif // CU_EXTERNS_CU_