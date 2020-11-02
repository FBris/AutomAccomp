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
 *  \file    ServerGPU.cu
 *  \brief   ReMAS for Nvidia GPUs USING Unified Memory
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
/* This include the first */
#include "../common/TimeFunctions.h"

#include <stdio.h>

extern "C" {
  #include "../common/FileFunctions.h"
}

__constant__ MyType CCosts[T_COSTS]; /* Placed within GPU Const Memory. Using by DTW */
__constant__ int  Ckmax_fft[N_MIDI]; /* Placed within GPU Const Memory. Using by FFT */
__constant__ int  Ckmin_fft[N_MIDI]; /* Placed within GPU Const Memory. Using by FFT */

#include "GPUFunctions.h"
#include "kernels.cuh"
#include "../common/SoundFunctions.h"
#include "../common/TempoFunctions.h"
#include "../common/NetFunctions.h"

/**
 *  \fn    int main(int argc , char *argv[])
 *  \brief main is the entry point to ReMAS, classical C/CUDA main program.
 *  \param argc: (in) As is usual in programs C/CUDA
 *  \param argv: (in) As is usual in programs C/CUDA
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int main (int argc , char *argv[])
{  
   /* for the GPU check, control and ... */
   int
     maxGrid,    TPBlock,
     GridNBASES, GridNFFTd2,
     GridNFFT,   GridNMID32,
     GridDTWSize;

   cudaEvent_t
     start, stop;

   cublasHandle_t
     handle;
            
   /* Using only by DTW */
   int
     pos_min,
     DTWWhere, DTWSize,
     DTWSizePlusPad;

   MyType
     *pD = NULL,
     Costs[T_COSTS];

   /* Using only by FFT */
   MyFFTGPUType
     plan;

   MyType
     *X_fft  =NULL,
     *Out_fft=NULL,
     *Mod_fft=NULL;
     
   int
     kmax_fft[N_MIDI],
     kmin_fft[N_MIDI],
     N_FFTdiv2;

   /* Using only by HMM (silence detection) */
   MyType
     prob_silen = (MyType)0.75,
     prob_audio = (MyType)0.25;

   bool
     Silence=true;

   /* Using by other preprocessing steps */
   MyType
     *norms     =NULL,
     *s_fk      =NULL,
     *v_SxD     =NULL,
     *v_cfreq   =NULL,
     *v_hanning =NULL,
     *v_dxState =NULL,
     *tauxi     =NULL,
     *ts_fk     =NULL,
     *sdata     =NULL,
     BETA=(MyType)1.5;

   int
     *states_time_i = NULL,
     *states_time_e = NULL,
     *states_seq    = NULL,
     *states_corr   = NULL,
     *I_SxD         = NULL;

   short
     *frame=NULL, *tmpframe=NULL;

   DTWconst  Param;
   DTWfiles  NameFiles;
   WAVHeader WavHeader;

   #ifdef ALSA
     snd_pcm_t         *SoundHandle=NULL;
     snd_pcm_uframes_t  SoundBufferSize;
   #endif

   /* Using by OSC for messages. Limited to MaxOSC (see defines.h) clients */
   #ifdef OSC
     lo_address DirOSC[MaxOSC];
   #endif

   /* For TEMPO */
   int      *preload=NULL;
   STempoRL TEMPORL;
   /* with nvcc we need to do using this way */
   TEMPORL.NextFrame=0;   TEMPORL.PrevState=0; TEMPORL.SynthSpeed=1.0;   TEMPORL.SoloSpeed=1.0; TEMPORL.numap=1;
   TEMPORL.SynthTime=0.0; TEMPORL.matched=1;   TEMPORL.AudioTimeAP[0]=0; TEMPORL.ScoreTimeAP[0]=0;

   /* For Docker and AppImage usage */
   bool UseOSC=false, UseMic=false;

   /* For time under CUDA domains */
   float time=0.0;

   /* General & others varibles */
   int i, j, NumTramas=0;
   FILE *fp=NULL;

   /* Reading global paramentes */
   switch(argc) {
      case 4: /* Regular use */
        #ifdef SIMPLE
          BETA=strtof(argv[1], NULL);
        #else
          BETA=strtod(argv[1], NULL);
        #endif
        TPBlock=atoi(argv[3]);
        #ifdef OSC
          UseOSC=true;
        #endif
        #ifdef ALSA
          UseMic=true;
        #endif
        break;
      case 6: /* For Docker or AppImagen */
        #ifdef SIMPLE
          BETA=strtof(argv[1], NULL);
        #else
          BETA=strtod(argv[1], NULL);
        #endif
        TPBlock=atoi(argv[3]);
        UseOSC =atoi(argv[4]);
        UseMic =atoi(argv[5]);
        break;
      default:
        printf("General usage: %s <BETA> <configuration file> <threadsPerBlock>\n", argv[0]);
        printf("   Example: %s 1.5 parametes.dat 64\n\n", argv[0]);
        printf("Docker  usage: %s <BETA> <configuration file> <threadsPerBlock> <OSC yes|no [1|0]> <alsa yes|no [1|0]>\n", argv[0]);
        printf("   Example: %s 1.5 parametes.dat 64 1 1\n\n", argv[0]);
        return -1;
   }

   /* Have we a compatible GPU? We assume that we only use one GPU (with ID = 0) */
   CHECKERR(HaveCompatibleGPU(maxGrid));
   
   #ifndef SIMPLE
     CUDAERR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
   #endif

   CUBLASERR(cublasCreate(&handle));
   CUDAERR(cudaEventCreate(&start));
   CUDAERR(cudaEventCreate(&stop));

   /* Reading general information and file names */
   CHECKERR(ReadParameters(&Param, &NameFiles, argv[2]));

   /* Allocating memory and reading some structures */
   CHECKERR(AllocDataGPU(&v_hanning, &states_time_i, &states_time_e, &states_seq, &states_corr, &I_SxD, &DTWSize, TAMTRAMA, Param.N_STATES, NameFiles));

   /* Allocating memory for and setting some DTW constants */
   DTWSizePlusPad =(DTWSize +  N_COSTS) * (N_COSTS + 1);
   CHECKERR(AllocDTWGPU(&pD, &v_SxD, &sdata, maxGrid, DTWSize, DTWSizePlusPad));

   /* Allocating memory for s_fk and auxiliar structures when Beta !=0.0 and !=1.0 */
   CHECKERR(AllocS_fkGPU(&s_fk, &tauxi, &ts_fk, BETA, N_MIDI_PAD, Param.N_BASES, NameFiles));

   /* Allocating memory for FFT memory and reading data */
   CHECKERR(AllocFFTGPU(&plan, &X_fft, &Out_fft, &Mod_fft, kmin_fft, kmax_fft, N_FFT, NameFiles));

   /* Allocating memory for the rest of general structures */
   CHECKERR(AllocAuxiGPU(&norms, &frame, &tmpframe, &v_cfreq, &v_dxState, Param.N_BASES, TAMTRAMA, N_MIDI));

   /* Initializing vector of costs and GPU const memory with this information */
   Costs[0] = 1.0; // Distance to (1,1)
   Costs[1] = 1.0; // Distance to (1,2)
   Costs[2] = 1.0; // Distance to (1,3)
   Costs[3] = 1.0; // Distance to (1,4)
   Costs[4] = 2.0; // Distance to (2,1)
   Costs[5] = 3.0; // Distance to (3,1)
   Costs[6] = 4.0; // Distance to (4,1)

   /* Moving info to GPU const memory */
   CUDAERR(cudaMemcpyToSymbol(Ckmax_fft, kmax_fft, sizeof(kmax_fft)));
   CUDAERR(cudaMemcpyToSymbol(Ckmin_fft, kmin_fft, sizeof(kmin_fft)));
   CUDAERR(cudaMemcpyToSymbol(CCosts,    Costs,    sizeof(Costs)));

   /* For TEMPO */
   CHECKNULL(preload=(int *)calloc(DTWSize, sizeof(int)));
   j=0;
   for (i=0; i<DTWSize; i++)
   {
     if (i > states_time_e[j]) j++;
     preload[i]=j;
   }

   /* Some helper constants */
   N_FFTdiv2    = N_FFT/2;
   GridNBASES   = (Param.N_BASES + TPBlock - 1) / TPBlock;
   GridNFFT     = (N_FFT         + TPBlock - 1) / TPBlock;   
   GridNFFTd2   = (N_FFTdiv2     + TPBlock)     / TPBlock;
   GridDTWSize  = (DTWSize       + TPBlock - 1) / TPBlock;
   GridNMID32   = (N_MIDI        + 31)          / 32;

   dim3 Grid2D((Param.N_BASES + 15) / 16, 1);
   dim3 TPBl2D(32,16);

   /* If OSC UPD is available */
   #ifdef OSC
   if (UseOSC)
     for (i=0; i<Param.NCliOSC; i++)
     {
       /* Declare an OSC destination, given IP address and port number, using UDP */
       CHECKERR(lo_address_errno(DirOSC[i]=lo_address_new(Param.HostIP[i], Param.HostPort[i])));
       #ifdef TALK
         printf("IPV4 %s, Port %s\n", Param.HostIP[i], Param.HostPort[i]);
       #endif
     }
   #endif

   /* Configure microphone input device if ALSA (Linux) or Core Audio (MacOS) is used */
   #ifdef ALSA
     if (UseMic)
     {
       SoundBufferSize=SetMicParams(&SoundHandle, Param);
       if (SoundBufferSize <=0) CHECKERR(ErrAlsaHw);

       #ifdef DUMP
         CHECKNULL(fp=fopen("FramesMicRecorded.pcm", "w"));
       #endif
       NumTramas = (Param.Time_MIC * AlsaRate) / TAMMUESTRA;
     }else{
       CHECKNULL(fp=fopen(NameFiles.file_frame, "rb"));

       CHECKERR(Read_WAVHeader(&WavHeader, fp));
       NumTramas = (WavHeader.num_samples - TAMTRAMA) / TAMMUESTRA;
     }
   #else
     CHECKNULL(fp=fopen(NameFiles.file_frame, "rb"));

     CHECKERR(Read_WAVHeader(&WavHeader, fp));
     NumTramas = (WavHeader.num_samples - TAMTRAMA) / TAMMUESTRA;
   #endif

   // Init DTW
   /* Where we start to store data within circular-buffer ?¿                       */
   /* if column 0 the 2nd parameter should be DTWSizePlusPad-DTWSize               */
   /* for other columns i*(DTWSize + N_COSTS)-DTWSize, where i is the column index */
   /* We start in column 1. Thereby i*(DTWSize + N_COSTS)-DTWSize=N_COSTS          */
   kernel_InitDTW<<<(DTWSizePlusPad+TPBlock-1)/TPBlock, TPBlock>>>(pD, N_COSTS, DTWSizePlusPad);

   // Compute norms */ 
   if (BETA>=0.0 && BETA<=0.0)
      kernel_CompNorB0<<<GridNBASES, TPBlock>>>(norms, (MyType)N_MIDI, Param.N_BASES);
   else if (BETA>=1.0 && BETA<=1.0)
      kernel_CompNorB1<<<Grid2D, TPBl2D>>>(norms, s_fk, N_MIDI, Param.N_BASES);
   else
      kernel_CompNorBG<<<Grid2D, TPBl2D>>>(norms, ts_fk, s_fk, N_MIDI, BETA, Param.N_BASES);
   
   /* Fill buffer */
   #ifdef ALSA
     if (UseMic)
       CHECKERR(ReadAlsaGPU1st(frame, tmpframe, SoundHandle, fp));
     else
       CHECKERR(ReadWavGPU1st (frame, tmpframe, fp));
   #else
     CHECKERR(ReadWavGPU1st (frame, tmpframe, fp));
   #endif

   #ifdef TALK
     printf("Listening ...\n");
   #endif

   /* Procedure for silence/white noise detection */
   while (Silence)
   {
     #ifdef ALSA
       if (UseMic)
         CHECKERR(ReadAlsaGPU(frame, tmpframe, SoundHandle, fp));
       else
         CHECKERR(ReadWavGPU(frame, tmpframe, fp));
     #else
       CHECKERR(ReadWavGPU(frame, tmpframe, fp));
     #endif
     NumTramas--;

     kernel_ApplyWindow<<<GridNFFT, TPBlock>>>(X_fft, frame, v_hanning, TAMTRAMA, N_FFT);

     CHECKERR(FFTGPU(X_fft, Out_fft, &plan));
     kernel_Modul<<<GridNFFTd2, TPBlock>>>(Mod_fft, Out_fft, N_FFTdiv2);
     kernel_Cfreq<<<N_MIDI, sizeWarp>>>(v_cfreq, Mod_fft);

     if (BETA>=0.0 && BETA<=0.0) {
       /* moving to BETA=1.0, not defined for BETA=0 This is an issue */
       kernel_Reduction<<<1, sizeWarp>>>(v_cfreq, N_MIDI);
       kernel_BetaNorm<<<1, N_MIDI>>>(v_cfreq, N_MIDI);
       kernel_CompDisB0<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, N_MIDI, Param.N_BASES);
     }
     else if (BETA>=1.0 && BETA<=1.0) {
       kernel_Reduction<<<1, sizeWarp>>>(v_cfreq, N_MIDI);
       kernel_BetaNorm <<<1, N_MIDI>>>(v_cfreq, N_MIDI);
       kernel_Reduction<<<1, sizeWarp>>>(v_cfreq, N_MIDI);
       kernel_CompDisB1<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, N_MIDI, Param.N_BASES);
     }
     else {
       kernel_ReductionPowBeta<<<1, sizeWarp>>>(v_cfreq, BETA, N_MIDI);
       kernel_BetaNorm <<<1, N_MIDI>>>(v_cfreq, N_MIDI);
       kernel_PowToReal<<<GridNMID32, sizeWarp>>>(tauxi, v_cfreq, BETA, N_MIDI);
       kernel_CompDisBG<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, ts_fk, tauxi, BETA, N_MIDI, Param.N_BASES);
     }
     cudaDeviceSynchronize();     
     Silence = DetectSilence((v_dxState[1]-v_dxState[0]), &prob_silen, &prob_audio);
   }

   /* Start OSC */
   #ifdef OSC
     if (UseOSC) for (i=0; i<Param.NCliOSC; i++) { CHECKERR(SendPlay(DirOSC[i])); CHECKERR(SendTempo(DirOSC[i], 110)); }
   #endif

   CUDAERR(cudaDeviceSynchronize());
   CUDAERR(cudaEventRecord(start, 0));

   /* start the system */
   for(i=1; i<=NumTramas; i++)
   {
      kernel_ApplyWindow<<<GridNFFT, TPBlock>>>(X_fft, frame, v_hanning, TAMTRAMA, N_FFT);

      CHECKERR(FFTGPU(X_fft, Out_fft, &plan));
      kernel_Modul<<<GridNFFTd2, TPBlock>>>(Mod_fft, Out_fft, N_FFTdiv2);
      kernel_Cfreq<<<N_MIDI, sizeWarp>>>(v_cfreq, Mod_fft);

      if (BETA>=0.0 && BETA<=0.0) {
        kernel_CompDisB0<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, N_MIDI, Param.N_BASES);
      }
      else if (BETA>=1.0 && BETA<=1.0) {
        kernel_Reduction<<<1, sizeWarp>>>(v_cfreq, N_MIDI);
        kernel_CompDisB1<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, N_MIDI, Param.N_BASES);
      }
      else {
        kernel_PowToReal<<<GridNMID32, sizeWarp>>>(tauxi, v_cfreq, BETA, N_MIDI);
        kernel_CompDisBG<<<Grid2D, TPBl2D>>>(v_dxState, v_cfreq, norms, s_fk, ts_fk, tauxi, BETA, N_MIDI, Param.N_BASES);
      }

      InitSxD(sdata, v_SxD, v_dxState, I_SxD, maxGrid, DTWSize);
      kernel_UpdateSxD<<<GridDTWSize, TPBlock>>>(v_SxD, Param.ALPHA, sdata, DTWSize);

      DTWWhere=(i % TBLOCK) * (N_COSTS + DTWSize) + N_COSTS;
      kernel_DTW<<<GridDTWSize, TPBlock>>>(v_SxD, pD, i, DTWWhere, DTWSize);
      #ifdef SIMPLE
        CUBLASERR(cublasIsamin(handle, DTWSize, &pD[DTWWhere], 1, &pos_min));
      #else
        CUBLASERR(cublasIdamin(handle, DTWSize, &pD[DTWWhere], 1, &pos_min));
      #endif
      pos_min--;

      /* Is it necessary to recalculate the tempo? (sec 2.4.1) */
      #ifdef OSC
        if (UseOSC)
          CHECKERR(ComputeTempoOSCRL(&TEMPORL, i, pos_min, preload[pos_min], states_corr, DirOSC, Param.NCliOSC));
        else
          ComputeTempoRL(&TEMPORL, i, pos_min, preload[pos_min], states_corr);
      #else
        ComputeTempoRL(&TEMPORL, i, pos_min, preload[pos_min], states_corr);
      #endif

      #ifndef ALSA
        /* waiting some milliseconds: simulating within the WAV file the audio delay */
        //CHECKERR(msleep(10));
      #endif

      // Read new data 
      #ifdef ALSA
        if (UseMic)
          CHECKERR(ReadAlsaGPU(frame, tmpframe, SoundHandle, fp));
        else
          CHECKERR(ReadWavGPU(frame, tmpframe, fp));
      #else
        CHECKERR(ReadWavGPU(frame, tmpframe, fp));
      #endif
   }
   CUDAERR(cudaEventRecord(stop, 0));
   CUDAERR(cudaEventSynchronize(stop));
   CUDAERR(cudaEventElapsedTime(&time, start, stop));

   printf("%f msec.\n", time);   

   /* Leave MS on tempo 1 and stop */
   #ifdef OSC
     if (UseOSC)
       for (i=0; i<Param.NCliOSC; i++) { CHECKERR(SendTempo(DirOSC[i], 100)); CHECKERR(SendPlay(DirOSC[i])); }
   #endif

   /* Close files and free sound device if used */
   FreeFiles(&NameFiles);
   #ifdef ALSA
     if (UseMic)
     {
       #ifdef DUMP
         fclose(fp);
       #endif
       CHECKERR(snd_pcm_close(SoundHandle));
     } else { fclose(fp); }
   #else
     fclose(fp);
   #endif

   /* frees in general. Is it necessary ? Not really but it is our habit */
   free(preload);
   free(states_corr);
   free(states_seq);
   free(states_time_i);
   free(states_time_e);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   CUBLASERR(cublasDestroy(handle));

   CUFFTERR(cufftDestroy(plan));

   if (!(BETA>=(MyType)0.0 && BETA<=(MyType)0.0) && !(BETA>=(MyType)1.0 && BETA<=(MyType)1.0)) {
      CUDAERR(cudaFree(tauxi));
      CUDAERR(cudaFree(ts_fk));
   }
   CUDAERR(cudaFree(I_SxD));
   CUDAERR(cudaFree(Mod_fft));
   CUDAERR(cudaFree(norms));
   CUDAERR(cudaFree(Out_fft));
   CUDAERR(cudaFree(pD));
   CUDAERR(cudaFree(sdata));
   CUDAERR(cudaFree(v_cfreq));
   CUDAERR(cudaFree(v_dxState));
   CUDAERR(cudaFree(v_hanning));
   CUDAERR(cudaFree(s_fk));
   CUDAERR(cudaFree(frame));
   CUDAERR(cudaFreeHost(tmpframe));
   CUDAERR(cudaFree(v_SxD));
   CUDAERR(cudaFree(X_fft));

   return OK;
}
