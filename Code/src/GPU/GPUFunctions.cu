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
 *  \file    GPUFunctions.cu
 *  \brief   File with code of ReMAS functions for Nvidia GPUs
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#include "GPUFunctions.h"

extern "C" {
#include "../common/FileFunctions.h"
}


/**
 *  \fn    unsigned int NextPow2(unsigned int x)
 *  \brief NextPow2 returns the next power of 2 of a given number
 *  \param x: (in) The number
 *  \return: Next power of 2 of a number
*/
unsigned int NextPow2(unsigned int x)
{
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}


/**
 *  \fn    inline bool IsPow2(unsigned int x)
 *  \brief IsPow2 decides if a number is power of 2 
 *  \param x: (in) The number
 *  \return: True is the number is power of 2, 0 otherwise false
*/
inline bool IsPow2(unsigned int x) { return ((x&(x-1))==0); }


/**
 *  \fn    int HaveCompatibleGPU(int &maxGrid)
 *  \brief HaveCompatibleGPU checks if the system has an appropiate GPU for ReMAS
 *  \param maxGrid: (out) MaxGrid stores the GPU maxGridSize property
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int HaveCompatibleGPU(int &maxGrid)
{
   int deviceCount, driverVersion;
   
   cudaDeviceProp deviceProp;

   CUDAERR(cudaGetDeviceCount(&deviceCount));

   CUDAERR(cudaGetDeviceProperties(&deviceProp, 0));
   if (deviceProp.major < 3) {
      printf("Sorry, we need CUDA Capability >=3\n");
      return ErrGpuWrong;
   }
   maxGrid=deviceProp.maxGridSize[0];
   
   CUDAERR(cudaDriverGetVersion(&driverVersion));
   if ((driverVersion/1000) < 6) {
      printf("Sorry, we need CUDA Version >=6\n");
      return ErrGpuWrong;
   }

   if (!deviceProp.unifiedAddressing) {
      printf("Your system does not support Unified Memory\n");
      return ErrGpuWrong;
   }

   return OK;
}


/**
 *  \fn    int AllocS_fkGPU(MyType **s_fk, MyType **tauxi, MyType **ts_fk, const MyType BETA, const int nmidi, const int nbases, DTWfiles NameFiles)
 *  \brief AllocS_fkGPU Allocates memory for S_fk vector, read its data from file and initializes other auxiliar vectors
 *  \param s_fk:     (out) s_fk vector
 *  \param tauxi:    (out) Auxiliar vector tauxi
 *  \param ts_fk:    (out) Auxiliar vector ts_fk
 *  \param BETA:      (in) BETA value
 *  \param nmidi:     (in) Number of midi notes
 *  \param nbases:    (in) Number of bases/combinations
 *  \param NameFiles: (in) Struct with the file names
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocS_fkGPU(MyType **s_fk, MyType **tauxi, MyType **ts_fk, const MyType BETA, const int nmidi,
                 const int nbases, DTWfiles NameFiles)
{
   CUDAERR(cudaMallocManaged((void **)s_fk, sizeof(MyType)*nmidi*nbases, cudaMemAttachGlobal));
   CHECKERR(ReadS_fk((*s_fk), nbases, NameFiles.file_partitura));
   
   if (!(BETA>=(MyType)0.0 && BETA<=(MyType)0.0) && !(BETA>=(MyType)1.0 && BETA<=(MyType)1.0))
   {
      CUDAERR(cudaMallocManaged((void **)tauxi, sizeof(MyType)*nmidi,        cudaMemAttachGlobal));
      CUDAERR(cudaMallocManaged((void **)ts_fk, sizeof(MyType)*nmidi*nbases, cudaMemAttachGlobal));
   }

   return OK;
}


/**
 *  \fn    int AllocDataGPU(MyType **v_hanning, int **states_time_i, int **states_time_e, int **states_seq, int **states_corr, int **I_SxD, int *DTWSize, const int tamtrama, const int nstates, DTWfiles NameFiles)
 *  \brief AllocDataGPU Allocates memory and initializes some structures reading info from files
 *  \param v_hanning:     (out) v_hanning vector
 *  \param states_time_i: (out) states_time_i vector, contains the start-time of each state in frames
 *  \param states_time_e: (out) states_time_e vector, contains the end-time of each state in frames
 *  \param states_seq:    (out) states_seq vector, contains the base/combination that is performed in each state
 *  \param states_corr:   (out) states_corr vector
 *  \param I_SxD:         (out) I_SxD vector
 *  \param DTWSize:        (in) Size of DTW vectors
 *  \param tamtrama:       (in) Size of frames in samples
 *  \param nstates:        (in) Number of states 
 *  \param NameFiles:      (in) Struct with the file names
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocDataGPU(MyType **v_hanning, int **states_time_i, int **states_time_e, int **states_seq, int **states_corr,
                 int **I_SxD, int *DTWSize, const int tamtrama, const int nstates, DTWfiles NameFiles)
{
   int i, j, pos;
   
   CHECKNULL((*states_time_i)=(int *)calloc(nstates, sizeof(int)));
   CHECKNULL((*states_time_e)=(int *)calloc(nstates, sizeof(int)));
   CHECKNULL((*states_seq)   =(int *)calloc(nstates, sizeof(int)));
   CHECKNULL((*states_corr)  =(int *)calloc(nstates, sizeof(int)));

   CHECKERR(ReadVectorInt64((*states_seq),    nstates, NameFiles.fileStates_seq));
   CHECKERR(ReadVectorInt64((*states_time_i), nstates, NameFiles.fileStates_Time_i));
   CHECKERR(ReadVectorInt64((*states_time_e), nstates, NameFiles.fileStates_Time_e));
   CHECKERR(ReadVectorInt64((*states_corr),   nstates, NameFiles.fileStates_corr));

   (*DTWSize)=(*states_time_e)[nstates - 1] + 1;

   CUDAERR(cudaMallocManaged((void **)I_SxD, sizeof(int)*(*DTWSize), cudaMemAttachGlobal));

   pos=0;
   for (i=0; i<nstates; i++)
   {
      for (j=(*states_time_i)[i]; j<=(*states_time_e)[i]; j++)
      {
         (*I_SxD)[pos]=(*states_seq)[i];
         pos++;
       }
   }

   CUDAERR(cudaMallocManaged((void **)v_hanning, sizeof(MyType)*tamtrama, cudaMemAttachGlobal));
   CHECKERR(ReadVector((*v_hanning), tamtrama, NameFiles.file_hanning));

   return OK;
}


/**
 *  \fn    int AllocFFTGPU(MyFFTGPUType *plan, MyType **X_fft, MyType **Out_fft, MyType **Mod_fft, int *kmin_fft, int *kmax_fft, const int nfft, DTWfiles NameFiles)
 *  \brief AllocFFTGPU Allocates "Unified" GPU memory for FFT vector and reads some fft information from files
 *  \param plan:     (out) FFT scheduler
 *  \param X_fft:    (out) X_fft vector
 *  \param Out_fft:  (out) Out_fft vector
 *  \param Mod_fft:  (out) Mod_fft vector
 *  \param kmin_fft: (out) Where kmin_fft is stored
 *  \param kmax_fft: (out) Where kmax_fft is stored
 *  \param nfft:      (in) As is to be, Mod_fft and Out_fft size 
 *  \param NameFiles: (in) Struct with the file names
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocFFTGPU(MyFFTGPUType *plan, MyType **X_fft, MyType **Out_fft, MyType **Mod_fft, int *kmin_fft,
                int *kmax_fft, const int nfft, DTWfiles NameFiles)
{
   CUDAERR(cudaMallocManaged((void **)X_fft,   sizeof(MyType)*2*nfft+1, cudaMemAttachGlobal));
   CUDAERR(cudaMallocManaged((void **)Mod_fft, sizeof(MyType)*nfft,     cudaMemAttachGlobal));
   /* ¿¿ works with Mod_fft  size=nfft/2+1 ?? */

   #ifdef SIMPLE
      CUDAERR(cudaMallocManaged((void **)Out_fft, sizeof(cufftComplex)*nfft, cudaMemAttachGlobal));
      CUFFTERR(cufftPlan1d(plan, nfft, CUFFT_R2C, 1));
   #else
      CUDAERR(cudaMallocManaged((void **)Out_fft, sizeof(cufftDoubleComplex)*nfft, cudaMemAttachGlobal));
      CUFFTERR(cufftPlan1d(plan, nfft, CUFFT_D2Z, 1));
   #endif

   if (plan==NULL) return ErrFFTSched;

   CHECKERR(ReadVectorInt64(kmax_fft, N_MIDI, NameFiles.file_kmax));
   CHECKERR(ReadVectorInt64(kmin_fft, N_MIDI, NameFiles.file_kmin));

   return OK;
}


/**
 *  \fn    int AllocDTWGPU(MyType **pV, MyType **v_SxD, MyType **sdata, const int maxGrid, const int DTWSize, const int DTWSizePlusPad)
 *  \brief AllocDTWGPU Allocates memory for DTW vectors and auxiliar structures
 *  \param pV:            (out) DTW pV vector
 *  \param v_SxD:         (out) v_SxD vector
 *  \param sdata:         (out) sdata vector, auxiliar
 *  \param maxGrid:        (in) maxGridSize supported by GPU
 *  \param DTWSize:        (in) Size of DTW vectors
 *  \param DTWSizePlusPad: (in) Size of DTW vectors plus padding
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocDTWGPU(MyType **pV, MyType **v_SxD, MyType **sdata, const int maxGrid, const int DTWSize, const int DTWSizePlusPad)
{
   int numThreads, numBlocks, sharedSize;

   BlocksAndThreads(&numBlocks, &numThreads, &sharedSize, maxGrid, DTWSize);

   CUDAERR(cudaMallocManaged((void **)pV,    sizeof(MyType)*DTWSizePlusPad, cudaMemAttachGlobal));
   CUDAERR(cudaMallocManaged((void **)v_SxD, sizeof(MyType)*DTWSize,        cudaMemAttachGlobal));
   CUDAERR(cudaMallocManaged((void **)sdata, sizeof(MyType)*numBlocks,      cudaMemAttachGlobal));

   return OK;
}


/**
 *  \fn    int AllocAuxiGPU(MyType **norms, short **GPUframe,  short **CPUframe, MyType **v_cfreq, MyType **v_dxState, const int nbases, const int tamframe, const int nmidi)
 *  \brief AllocAuxiGPU memory reservation for norms, frame, v_cfreq and v_dxState vectors
 *  \param norms:     (out) Norms vector
 *  \param GPUframe:  (out) Vector for frames in GPU memory
 *  \param CPUframe:  (out) Vector for frames in CPU memory
 *  \param v_cfreq:   (out) v_cfreq vector
 *  \param v_dxState: (out) v_dxState vector
 *  \param nbases:     (in) Number of bases/combinations, sizeof norms and v_dxState
 *  \param tamframe    (in) Size of frames in samples
 *  \param nmidi:      (in) Number of midi notes, v_cfreq size
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocAuxiGPU(MyType **norms, short **GPUframe, short **CPUframe, MyType **v_cfreq, MyType **v_dxState, const int nbases,
                 const int tamframe, const int nmidi)
{
   CUDAERR(cudaMallocManaged((void **)norms,     sizeof(MyType)*nbases,  cudaMemAttachGlobal));
   CUDAERR(cudaMallocManaged((void **)v_dxState, sizeof(MyType)*nbases,  cudaMemAttachGlobal));
   CUDAERR(cudaMallocManaged((void **)v_cfreq,   sizeof(MyType)*nmidi,   cudaMemAttachGlobal));

   CUDAERR(cudaMalloc   ((void **)GPUframe, sizeof(short)*tamframe));
   CUDAERR(cudaHostAlloc((void **)CPUframe, sizeof(short)*tamframe, cudaHostAllocWriteCombined));

   return OK;
}


/**
 *  \fn    void BlocksAndThreads(int *blocks, int *threads, int *sharedsize, const int maxGrid, const int size)
 *  \brief BlocksAndThreads calculates the suitable number of blocks and threads, and the needed shared memory
 *  \param blocks:     (out) Number of blocks
 *  \param threads:    (out) Number of threads
 *  \param sharedsize: (out) Size of shared memory
 *  \param maxGrid:     (in) maxGridSize supported by GPU
 *  \param size:        (in) Size of the vectors
 *  \return: Nothing, it is void
*/
void BlocksAndThreads(int *blocks, int *threads, int *sharedsize, const int maxGrid, const int size)
{
   (*threads) = (size < maxThreads*2) ? NextPow2((size + 1)/ 2) : maxThreads;
   (*blocks)  = (size + ((*threads) * 2 - 1)) / ((*threads) * 2);

   if ((*blocks) > maxGrid)
   {
      (*blocks)  /= 2;
      (*threads) *= 2;
   }

   (*blocks)     = min(maxBlocks, (*blocks));
   (*sharedsize) = ((*threads) <= 32) ? 2*(*threads)*sizeof(MyType) : (*threads)*sizeof(MyType);
}


/** 
 *  \fn    int FFTGPU(MyType *X_fft, MyType *Out_fft, MyFFTGPUType *plan)
 *  \brief FFTGPU computes FFT
 *  \param X_fft:   (inout) X_fft vector
 *  \param Out_fft: (inout) Out_fft vector
 *  \param plan:      (out) FFT scheduler
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int FFTGPU(MyType *X_fft, MyType *Out_fft, MyFFTGPUType *plan)
{   
   #ifdef SIMPLE
      CUFFTERR(cufftExecR2C(*plan, (cufftReal *)X_fft,       (cufftComplex *)Out_fft));
   #else
      CUFFTERR(cufftExecD2Z(*plan, (cufftDoubleReal *)X_fft, (cufftDoubleComplex *)Out_fft));
   #endif

   return OK;
}


/** 
 *  \fn    void InitSxD(MyType *odata, MyType *v_SxD, const MyType* __restrict__ v_dxState, const int* __restrict__ I_SxD, const int maxGrid, const int size)
 *  \brief InitSxD launches the cuda kernel that sets up the vector SxD when "Unified" GPU memory is used.
 *  \param odata:   (inout) Intermedial data vector
 *  \param v_SxD:     (out) v_SxD vector
 *  \param v_dxState: (out) v_dxState vector
 *  \param I_SxD:     (out) I_SxD vector
 *  \param maxGrid:    (in) maxGridSize supported by GPU
 *  \param size:       (in) Size of the vector
 *  \return: The pos of "a" minimum
*/
void InitSxD(MyType *odata, MyType *v_SxD, const MyType* __restrict__ v_dxState, const int* __restrict__ I_SxD,
             const int maxGrid, const int size)
{
   int numBlocks=0, numThreads=0, sharedSize=0, s;

   BlocksAndThreads(&numBlocks, &numThreads, &sharedSize, maxGrid, size);

   kernel_InitSxD<<<numBlocks, numThreads, sharedSize>>>(odata, v_SxD, v_dxState, I_SxD, numThreads, IsPow2(size), size);

   s = numBlocks;
   while (s > 1)
   {
      BlocksAndThreads(&numBlocks, &numThreads, &sharedSize, maxGrid, s);

      kernel_Sum<<<numBlocks, numThreads, sharedSize>>>(odata, odata, numThreads, IsPow2(s), s);
      s = (s + (numThreads*2-1)) / (numThreads*2);
 
   }
   kernel_Vnorm<<<1, 1>>>(odata);
}


/**
  *  \fn    ReadWavGPU1st(short *GPUframe, short *CPUframe, FILE *fp)
  *  \brief ReadWavGPU1st reads first audio (frame) from WAV file when NVIDIA GPU is used
  *  \param GPUframe: (out) Vector to store the first frame within NVIDIA GPU
  *  \param CPUframe: (---) Vector to store the first frame within ARM
  *  \param fp:        (in) ID of file with the information
  *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadWavGPU1st(short *GPUframe, short *CPUframe, FILE *fp)
{
  if (fread(&CPUframe[TAMMUESTRA], sizeof(short), TTminusTM, fp) != TTminusTM) return ErrReadFile;

  CUDAERR(cudaMemcpy(&GPUframe[TAMMUESTRA], &CPUframe[TAMMUESTRA], sizeof(short)*TTminusTM, cudaMemcpyHostToDevice));
      
  return OK;
}

/**
  *  \fn    ReadWavGPU(short *GPUframe, short *CPUframe, FILE *fp)
  *  \brief ReadFileGPU reads current audio (frame) from WAV file when NVIDIA GPU is used
  *  \param GPUframe: (out) Vector to store the current frame whitin NVIDIA GPU
  *  \param CPUframe: (---) Vector to store the current frame whitin ARM
  *  \param fp:        (in) ID of file with the information
  *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadWavGPU(short *GPUframe, short *CPUframe, FILE *fp)
{
  kernel_Shift<<<1, TAMMUESTRA>>>(GPUframe, TAMTRAMA, TAMMUESTRA);

  if (fread(CPUframe, sizeof(short), TAMMUESTRA, fp) != TAMMUESTRA) return ErrReadFile;

  // ¿¿ cudaDeviceSynchronize(); ??

  CUDAERR(cudaMemcpy(&GPUframe[TTminusTM], CPUframe, sizeof(short)*TAMMUESTRA, cudaMemcpyHostToDevice));

  return OK;
}


#ifdef ALSA
  /**
   *  \fn    int ReadAlsaGPU1st(short *GPUframe, short *CPUframe, snd_pcm_t *DeviceID, FILE *fpdump)
   *  \brief ReadAlsaGPU1st reads from microphone the first audio (frame) when GPU is used
   *  \param GPUframe: (out) Vector to store the first frame within NVIDIA GPU
   *  \param CPUframe: (---) Vector to store the first frame within ARM
   *  \param DeviceID:  (in) ALSA Sound devide identifier
   *  \param fpdump:   (out) File handle when DUMP is active
   *  \return: 0 if all is OK, otherwise a code error (see defines.h)
   * 
  */
  int ReadAlsaGPU1st(short *GPUframe, short *CPUframe, snd_pcm_t *DeviceID, FILE *fpdump)
  {
    if (snd_pcm_readi(DeviceID, &CPUframe[TAMMUESTRA], TTminusTM) != TTminusTM) return ErrReadDevice;

    CUDAERR(cudaMemcpy(&GPUframe[TAMMUESTRA], &CPUframe[TAMMUESTRA], sizeof(short)*TTminusTM, cudaMemcpyHostToDevice));

    #ifdef DUMP
      if (fwrite(&CPUframe[TAMMUESTRA], sizeof(short), TTminusTM, fpdump) != TTminusTM) return ErrWriteFile;
    #endif

    return OK;
  }
  
  /**
   *  \fn    int ReadAlsaGPU(short *GPUframe, short *CPUframe, snd_pcm_t *DeviceID, FILE *fpdump)
   *  \brief ReadAlsaGPU reads from microphone the current audio (frame) when GPU is used
   *  \param GPUframe: (out) Vector to store the current frame within NVIDIA GPU
   *  \param CPUframe: (---) Vector to store the current frame within ARM
   *  \param DeviceID:    (in) ALSA Sound devide identifier
   *  \param fpdump:     (out) File handle when DUMP is active
   *  \return: 0 if all is OK, otherwise a code error (see defines.h)
   * 
  */
  int ReadAlsaGPU(short *GPUframe, short *CPUframe, snd_pcm_t *DeviceID, FILE *fpdump)
  {
    kernel_Shift<<<1, TAMMUESTRA>>>(GPUframe, TAMTRAMA, TAMMUESTRA);

    if (snd_pcm_readi(DeviceID, CPUframe, TAMMUESTRA) != TAMMUESTRA) return ErrReadDevice;

    // ¿¿ cudaDeviceSynchronize(); ??

    CUDAERR(cudaMemcpy(&GPUframe[TTminusTM], CPUframe, sizeof(short)*TAMMUESTRA, cudaMemcpyHostToDevice));

    #ifdef DUMP
      if (fwrite(&CPUframe[TTminusTM], sizeof(short), TAMMUESTRA, fpdump) != TAMMUESTRA) return ErrWriteFile;
    #endif

    return OK;
  }
#endif
