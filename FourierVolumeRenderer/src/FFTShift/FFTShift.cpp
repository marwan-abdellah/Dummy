#include "FFTShift.h"
#include "shared.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

float ***iArr, ***oArr;

void FFTShift::prepareArrays(int N)
{
    iArr = (float***) malloc(N * sizeof(float**));
    for (int y = 0; y < N; y++)
    {
        iArr[y] = (float**) malloc (N * sizeof(float*));
        for (int x = 0; x < N; x++)
        {
            iArr[y][x] = (float*) malloc( N * sizeof(float));
        }
    }

    oArr = (float***) malloc(N * sizeof(float**));
    for (int y = 0; y < N; y++)
    {
        oArr[y] = (float**) malloc (N * sizeof(float*));
        for (int x = 0; x < N; x++)
        {
            oArr[y][x] = (float*) malloc( N * sizeof(float));
        }
    }
}

float** FFTShift::FFT_Shift_2D(float** iArr, float** oArr, int N)
{
    LOG();

    for (int i = 0; i < N/2; i++)
        for(int j = 0; j < N/2; j++)
        {
            oArr[(N/2) + i][(N/2) + j] = iArr[i][j];
            oArr[i][j] = iArr[(N/2) + i][(N/2) + j];

            oArr[i][(N/2) + j] = iArr[(N/2) + i][j];
            oArr[(N/2) + i][j] = iArr[i][(N/2) + j];
        }

    return oArr;
}

float* FFTShift::Repack_2D(float** Input_2D, float* Input_1D, int N)
{
    int ctr = 0;
    for (int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            Input_1D[ctr] = Input_2D[i][j];
            ctr++;
        }

    return Input_1D;
}

float*** FFTShift::FFT_Shift_3D(float* Input, int N)
{
    int ctr = 0;
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
            {
                iArr[i][j][k] = Input[ctr];
                oArr[i][j][k] = 0;
                ctr++;
            }

    for (int k = 0; k < N/2; k++)
        for (int i = 0; i < N/2; i++)
            for(int j = 0; j < N/2; j++)
            {
                oArr[(N/2) + i][(N/2) + j][(N/2) + k] = iArr[i][j][k];
                oArr[i][j][k] = iArr[(N/2) + i][(N/2) + j][(N/2) + k];

                oArr[(N/2) + i][j][(N/2) + k] = iArr[i][(N/2) + j][k];
                oArr[i][(N/2) + j][k] = iArr[(N/2) + i][j][(N/2) + k];

                oArr[i][(N/2) + j][(N/2) + k] = iArr[(N/2) + i][j][k];
                oArr[(N/2) + i][j][k] = iArr[i][(N/2) + j][(N/2) + k];


                oArr[i][j][(N/2) + k] = iArr[(N/2) + i][(N/2) + j][k];
                oArr[(N/2) + i][(N/2) + j][k] = iArr[i][j][(N/2) + k];
            }

    return oArr;
}

float* FFTShift::Repack_3D(float*** Input_3D, float* Input_1D, int N)
{
    int ctr = 0;
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
            {
                Input_1D[ctr] = Input_3D[i][j][k];
                ctr++;
            }

    return Input_1D;
}

float* FFTShift::flatFFT_Shift_3D(float* inputArray, int N)
{
    float* outputArray;

    outputArray = (float*) malloc (sizeof(float) * N * N * N);

    // Size of Single Line (1D)
    int sLine = N;

    // Slice Size (2D)
    int sSlice = N * N;

    // Volume Size (3D)
    int sVolume = N * N * N;

    // Transformation Equations
    int fTransEq1 = ((sVolume + sSlice + sLine)/2);
    int fTransEq2 = ((sVolume + sSlice)/2);
    int fTransEq3 = ((sVolume + sLine)/2);
    int fTransEq4 = ((sVolume/2));

    // Array Index
    int index = 0;

    // For Each Slice in the First Chunk (This Chunk Represents Half the Size of the 3D Volume)
    for (int slice = 0; slice < N/2; slice ++)
    {
        // For Each Line in the Slice
        for (int line = 0; line < N/2; line++)
        {
            // For Each Element in the Line
            for (int element = 0; element < N/2; element++)
            {
                // First Quad
                outputArray[index] = inputArray[fTransEq1 + index];

                // Second Quad
                outputArray[index + sLine/2] = inputArray[fTransEq2 + index];

                // Third Quad
                outputArray[index + sSlice/2] = inputArray[fTransEq3 + index];

                // Fourth Quad
                outputArray[index + sSlice/2 + sLine/2] = inputArray[fTransEq4 + index];

                // Increment Element Index per Line
                index++;
            }

            // Go to Next Line
            index = index + sLine/2;
        }

        // Go to Next Slice
        index = index + (sSlice/2);
    }

    // Process the Second Chunk (Half) of the 3D Volume
    // Reset the Index to Follow the New Equations
    index = 0;

    // For Each Slice in the Chunk
    for (int slice = 0; slice < N/2; slice ++)
    {
        // For Each Line in the Slice
        for (int line = 0; line < N/2; line++)
        {
            // For Each Element in the Line
            for (int element = 0; element < N/2; element++)
            {
                // First Quad
                outputArray[fTransEq1 + index] = inputArray[index];

                // Second Quad
                outputArray[fTransEq2 + index] = inputArray[index + sLine/2];

                // Third Quad
                outputArray[fTransEq3 + index] = inputArray[index + sSlice/2];

                // Fourth Quad
                outputArray[fTransEq4 + index] = inputArray[index + sSlice/2 + sLine/2];

                // Increment Element Index per Line
                index++;
            }

            // Go to Next Line
            index = index + sLine/2;
        }

        // Go to Next Slice
        index = index + (sSlice/2);
    }

    return outputArray;

}

