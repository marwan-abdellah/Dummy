#include "FFTShift.h"
#include "shared.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

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
    float ***iArr, ***oArr;;

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

    delete [] iArr;
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
