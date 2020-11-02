/**************************************************************************
 *   Copyright (C) 2017 by "Information Retrieval and Parallel Computing" *
 *   group (University of Oviedo, Spain), "Interdisciplinary Computation  *
 *   and Communication" group (Polytechnic University of Valencia, Spain) *
 *   and "Signal Processing and Telecommunication Systems Research" group *
 *   (University of Jaen, Spain)                                          *
 *   Contact: remaspack@gmail.com                                         *
 *                                                                        *
 *   This program is free software; you can redistribute it and/or modify *
 *   it under the terms of the GNU General Public License as published by *
 *   the Free Software Foundation; either version 2 of the License, or    *
 *   (at your option) any later version.                                  *
 *                                                                        *
 *   This program is distributed in the hope that it will be useful,      *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
 *   GNU General Public License for more details.                         *
 *                                                                        *
 *   You should have received a copy of the GNU General Public License    *
 *   along with this program; if not, write to the                        *
 *   Free Software Foundation, Inc.,                                      *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.            *
 **************************************************************************
*/
/**
 *  \file    GPUFunctions.h
 *  \brief   Header file for using ReMAS with Nvidia GPUs
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#pragma once

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* CUDA-C includes */
#include <cuda_runtime.h>

/* CuFFT includes */
#include <cufft.h>

/* CuBLAS includes */
#include <cublas_v2.h>

#ifdef ALSA
  #include <asoundlib.h>
#endif
  
#include "../common/defines.h"


/* ******************************** Preproceso Functions Prototypes  **************************** */
void         BlocksAndThreads(int*, int*, int*, const int, const int);
int          HaveCompatibleGPU(int &);
inline bool  IsPow2(unsigned int);
unsigned int NextPow2(unsigned int);


int FFTGPU(MyType*, MyType*, MyFFTGPUType*);

void InitSxD(MyType *, MyType *, const MyType* __restrict__, const int* __restrict__, const int, const int);

int AllocAuxiGPU(MyType **, short **, short **, MyType **, MyType **, const int, const int, const int);

int AllocDataGPU(MyType **, int **, int **, int **, int **, int **, int *, const int, const int, DTWfiles);

int AllocDTWGPU(MyType **, MyType **, MyType **, const int, const int, const int);

int AllocFFTGPU(MyFFTGPUType *, MyType **, MyType **, MyType **, int*, int*, const int, DTWfiles);

int AllocS_fkGPU(MyType **, MyType **, MyType **, const MyType, const int, const int, DTWfiles);

int ReadWavGPU1st(short *, short *, FILE *);
int ReadWavGPU   (short *, short *, FILE *);
#ifdef ALSA
  int ReadAlsaGPU1st(short *, short *, snd_pcm_t *, FILE *);
  int ReadAlsaGPU   (short *, short *, snd_pcm_t *, FILE *);
#endif

/* ********************************* Preproceso kernels Prototypes  ***************************** */
__global__ void kernel_ApplyWindow(MyType* __restrict__, const  short* __restrict__, const MyType* __restrict__, const int, const int);

__global__ void kernel_InitDTW(MyType* __restrict__, const int, const int);

__global__ void kernel_CompNorB0(MyType* __restrict__, const MyType, const int);

__global__ void kernel_CompNorB1(MyType* __restrict__, const MyType* __restrict__, const int, const int);

__global__ void kernel_CompNorBG(MyType* __restrict__, MyType* __restrict__, const MyType* __restrict__, const int, const MyType, const int);

__global__ void kernel_PowToReal(MyType* __restrict__, const MyType* __restrict__, const MyType, const int);

__global__ void kernel_Cfreq(MyType* __restrict__, const MyType* __restrict__);

__global__ void kernel_Modul(MyType* __restrict__, const MyType* __restrict__, const int);

__global__ void kernel_Reduction(MyType* __restrict__, const int);

__global__ void kernel_InitSxD(MyType* __restrict__, MyType* __restrict__, const MyType* __restrict__,
                               const int* __restrict__, const int, const bool, const int);

__global__ void kernel_Sum(MyType* __restrict__, const MyType* __restrict__, const int, const bool, const int);

__global__ void kernel_Vnorm(MyType* __restrict__);

__global__ void kernel_UpdateSxD(MyType* __restrict__, const MyType,  const MyType* __restrict__, const int);

__global__ void kernel_DTW(const MyType* __restrict__, MyType* __restrict__, MyType* __restrict__, int* __restrict__,
                           const int, const int, const int);

__global__ void kernel_CompDisB0(MyType* __restrict__, const MyType* __restrict__, const MyType* __restrict__,
                                 const MyType* __restrict__, const int, const int);

__global__ void kernel_CompDisB1(MyType* __restrict__, const MyType* __restrict__, const MyType* __restrict__,
                                 const MyType* __restrict__, const int, const int);

__global__ void kernel_CompDisBG(MyType* __restrict__, const MyType* __restrict__, const MyType* __restrict__,
                                 const MyType* __restrict__, const MyType* __restrict__, const MyType* __restrict__,
                                 const MyType, const int, const int);

__global__ void kernel_Shift(short* __restrict__, const int, const int);

#endif
