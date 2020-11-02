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
 *  \file    CPUFunctions.h
 *  \brief   Header file for using ReMAS with CPU, both x86_64 and ARM.
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
#include <unistd.h>
#include <stdbool.h> /* For docker and AppImage versions */

#ifdef OMP
  #include <omp.h>
#endif

#ifdef MKL
  #ifdef CBLAS
    #include <mkl.h>
  #endif
#else
  #ifdef CBLAS
    #include <cblas.h>
  #endif
#endif

#include <fftw3.h>

#ifdef ALSA
  #include <asoundlib.h>
#endif

#include "../common/defines.h"

int AllocDataCPU(MyType **, int **, int **, int **, int **, int **, int *, const int, const int, DTWfiles);

int AllocDTWCPU(MyType **, MyType **, const int, const int, const int);

int AllocSVelCPU(const int *, const int *, const int, const int, const int, SVelStates *);

int AllocS_fkCPU(MyType **, MyType **, MyType **, const MyType, const int, const int, DTWfiles);

int AllocFFTCPU(MyFFTCPUType *, MyType **, MyType **, MyType **, int*, int*, const int, DTWfiles);

int AllocAuxiCPU(MyType **, short **, MyType **, MyType **, const int, const int, const int);

void ApplyWindow(MyType * __restrict, const  short *, const MyType *,  const int, const int);

void ComputeNorms(MyType *, MyType *, const MyType *, const int, const int, const MyType);

void FFT(MyType *, const int *, const int *, MyType *, MyType *, MyType *, MyFFTCPUType, const int, const int);

void WriteAlignment(const int, const int, int *, const int *, FILE *);


int DTWProc(const MyType *, const MyType *, MyType * __restrict, const int, const int, const int);

int DTWProcVar(const MyType *, const MyType *, MyType * __restrict, const int, const int, const int);

int DTWProcMed(const MyType *, const MyType *, MyType * __restrict, const int, const int, const int,
               const int *, const int *, SVelStates *, const MyType);

void ComputeVelMed(MyType *, int, MyType *, MyType *, int);

void ApplyDist(const MyType *, MyType *, const int *, const int, const MyType);

void ComputeDist(const MyType *, MyType * __restrict, MyType *, const MyType *, const MyType *, const MyType *,
                 const MyType, const int, const int);

int ReadWavCPU1st(short *, FILE *);
int ReadWavCPU   (short *, FILE *);

#ifdef ALSA
  int ReadAlsaCPU1st(short *, snd_pcm_t *, FILE *);
  int ReadAlsaCPU   (short *, snd_pcm_t *, FILE *);
#endif

int SeqIdamin(const int, const MyType *);

#ifdef OMP
  int ParIdamin(const int, const MyType *);
#endif

#endif

void BetaNorm(MyType *, const int, MyType);
void BetaNormBien(MyType *, const int, MyType);
