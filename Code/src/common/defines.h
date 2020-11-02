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
 *  \file    defines.h
 *  \brief   General header file with constants, defines, structs, etc. using by ReMAS, both CPU and GPU.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/

#pragma once


#ifndef DATA_H
#define DATA_H


/* Problem constants */
#define N_MIDI     114 /* For now, N_MIDI     = 114             */
#define N_MIDI_PAD 128 /* Para que esté alineado                */
#define N_FFT    16384 /* For now, N_FFT      = 16384           */
#define TAMTRAMA  5700 /* For now, TAMTRAMA   = 10*TAMMUESTRA   */
#define TAMMUESTRA 570 /* For now, TAMMUESTRA = 570             */
#define TTminusTM (TAMTRAMA-TAMMUESTRA)

#define N_COSTS            4  /* For now 2 dimensins and 4 costs per dimension    */
#define T_COSTS (2*N_COSTS-1) /* Thereby total number of costs used is this value */
#define TBLOCK    (N_COSTS+1) /* another used derivate value                      */

#define Scaling 0.000030517578125 /* This is 1/32768, scaling factor for conversion */

/* GPU constants */
#define sizeWarp     32
#define maxThreads  512
#define maxBlocks   256

/* Local types */
#ifdef SIMPLE
   #define MyType       float
   #define MyFFTCPUType fftwf_plan
#else
   #define MyType       double
   #define MyFFTCPUType fftw_plan
#endif
#define MyFFTGPUType cufftHandle

#define WAVHeaderLength 44

/* ALSA sound parameters configuration */
#ifdef ALSA
  #define AlsaAccessMode   SND_PCM_ACCESS_RW_INTERLEAVED /* Value 3 see /usr/include/alsa/pcm.h) */
  #define AlsaChannels	   1                             /* Number of channels 1=mono */
  #define AlsaRate         44100                         /* Rate */
  #define AlsaAccessFormat SND_PCM_FORMAT_S16_LE         /* Signed 16 bit little endian */
#endif

/* Apple Audio Queue parameters configuration */
#ifdef CAUDIO
  #define AQRate               44100
  #define AQBufferSize         TAMMUESTRA
  #define AQChannels           1
  #define AQBitsPerChannel     16
  #define kNumberRecordBuffers 4
  #define MiddleBufferSize     (TAMTRAMA*2)
#endif

/* OSC */
#define MaxOSC 5 /* Maximum number of OSC clients supported */

/* Tempo */
#define RunJumpTime   0.012925                  /* 0.012925=(TAMMUESTRA / {AlsaRate or AQRate}), seconds */
#define TrainJumpTime 0.012925                  /* This depends on traning, seconds                      */
#define DelayTimeMidi 1                         /* Delay midi time for matching                          */
#define NUMAPMAX      12                        /* Number of anchor points for tempo estimation          */
#define NUMAPPAIRS    (NUMAPMAX*(NUMAPMAX-1)/2) /* Number of AP pairs for tempo estimation               */

/* Silence detection */
#define HMMchange 0.20 /* change state probability */
#define HMMremain 0.80 /* 1-HMMchange */

/* For check errors */
#define OK              0
#define ErrReadFile    -1
#define ErrWriteFile   -2
#define ErrInfoReaded  -3
#define ErrGpuWrong    -4
#define ErrFFTSched    -5
#define ErrReadDevice  -6
#define ErrSendOSC     -7
#define ErrAlsaHw      -8
#define ErrWavFormat   -9
#define ErrNULL       -10;
#define ErrTimer      -11;

#define CHECKERR(x) do { if((x)<0) { \
   printf("Error %d calling %s line %d\n", x, __FILE__, __LINE__);\
   return x;}} while(0)

#define CHECKNULL(x) do { if((x)==NULL) { \
   printf("NULL (when open file or  memory allocation calling %s line %d\n", __FILE__, __LINE__);\
   return ErrNULL;}} while(0)

#define CUDAERR(x) do { if((x)!=cudaSuccess) { \
   printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#define CUBLASERR(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
   printf("CUBLAS error: %s, line %d\n", __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#define CUFFTERR(x) do { if((x)!=CUFFT_SUCCESS) { \
   printf("CUFFT error: %s, line %d\n", __FILE__, __LINE__);\
   return EXIT_FAILURE;}} while(0)

#ifndef min
   #define min(x,y) ((x < y) ? x : y)
#endif

#ifdef OMP4
  #include <float.h>

  /** @struct PosMin
   *  Struct for store the minimum and its position
  */
  struct PosMin {
    MyType val;
    int    pos;
  };
  typedef struct PosMin PosMin;

  #ifdef SIMPLE
    #pragma omp declare reduction(MIN : PosMin : \
      omp_out = ((omp_in.val < omp_out.val) || ((omp_in.val == omp_out.val) && (omp_in.pos < omp_out.pos))) ? omp_in : omp_out) \
      initializer(omp_priv = { FLT_MAX, 0 })
  #else
    #pragma omp declare reduction(MIN : PosMin : \
      omp_out = ((omp_in.val < omp_out.val) || ((omp_in.val == omp_out.val) && (omp_in.pos < omp_out.pos))) ? omp_in : omp_out) \
      initializer(omp_priv = { DBL_MAX, 0 })
  #endif
#endif

/**
 *  \struct DTWconst
 *  \brief Struct for store global information of the problem.<BR>
 *  Each composition needs a file with values for these parameters.
 *  The structure of this text file, shared with DTWfiles struct
 *  should be one line per parameter. No empty lines and one empty
 *  line to the end. Example of this file:<BR>
 *  57                               // Number of bases/combinations.<BR>
 *  77                               // Number of states.<BR>
 *  0.15			     // Note duration (sec.) in the score.<BR>
 *  11.5129                          // ALPHA value.<BR>
 *  Datos/30sec/Input/hanning.bin    // Name and path to the file with hanning data.<BR>
 *  ...<BR>
 *  Datos/30sec/Input/corrstates.bin // Name and path to the file with corr_states data.<BR>
 *  1			             // If 0 input audio from WAV file. If 1 input audio from sound device.<BR>
 *  hw:2,0			     // Sound devide (mic) identification. If previous parameter=0 not used.<BR>
 *  150				     // MIDI time. Not used when ...<BR>
 *  5			             // Number of OSC clients. Limited to constant MaxOSC.<BR>
 *  127.0.0.1	       		     // If number of OSC client >0 the Host IP for the 1st client.<BR>
 *  5432			     // If number of OSC client >0 the Port    for the 1st client.<BR>
 *  ...<BR>
 *  127.0.0.5	       		     // If number of OSC client >0 the Host IP for the last client.<BR>
 *  5445			     // If number of OSC client >0 the Port    for the last client.<BR>
*/
struct DTWconst
{
   int    N_BASES;           /**< Number of bases/combinations. No default value.             */
   int    N_STATES;          /**< Number of states. No default value.                         */
   MyType NoteLen;           /**< Duration of the notes (sec.) in the score                   */
   MyType ALPHA;             /**< ALPHA parameter. Default value=11.5129                      */
   int    WAVorMIC;          /**< Equal to 0 input audio from WAV file, 1 from sound device   */
   char SoundID[32];         /**< Sound device (mic) identification. Used if WAVorMIC != 0    */
   int    Time_MIC;          /**< Sound time mic. Used if WAVorMIC != 0                       */
   int    NCliOSC;           /**< Number of OSC Clients. No default value                     */
   char  HostIP[MaxOSC][16]; /**< Host IP for OSC messages. Limited to 5. Used if NCLIOSC!= 0 */
   char HostPort[MaxOSC][5]; /**< Host port for OSC messages. Used if NCliOSC != 0            */
};

/**
 *  \struct DTWfiles
 *  \brief Struct for store the name of input/verificaton files.<BR>
 *  Each composition needs a file with values for these parameters.
 *  The structure of this text file, shared with DTWconst struct
 *  should be one line per parameter. No empty lines and one empty
 *  line to the end. Example of this file:<BR>
 *  57                               // Number of bases/combinations.<BR>
 *  77                               // Number of states.<BR>
 *  0.15			     // Note duration (sec.) in the score.<BR>
 *  11.5129                          // ALPHA value.<BR>
 *  Datos/30sec/Input/hanning.bin    // Name and path to the file with hanning data.<BR>
 *  ...<BR>
 *  Datos/30sec/Input/corrstates.bin // Name and path to the file with corr_states data.<BR>
 *  1			             // If 0 input audio from WAV file. If 1 input audio from sound device.<BR>
 *  hw:2,0			     // Sound devide (mic) identification. If previous parameter=0 not used.<BR>
 *  150				     // MIDI time. Not used when ...<BR>
 *  5			             // Number of OSC clients. Limited to constant MaxOSC.<BR>
 *  127.0.0.1	       		     // If number of OSC client >0 the Host IP for the 1st client.<BR>
 *  5432			     // If number of OSC client >0 the Port    for the 1st client.<BR>
 *  ...<BR>
 *  127.0.0.5	       		     // If number of OSC client >0 the Host IP for the last client.<BR>
 *  5445			     // If number of OSC client >0 the Port    for the last client.<BR>
*/
struct DTWfiles
{
   /* Files with input data for the problem */
   char *file_hanning;            /**< Name of the file with vector hanning data.       */
   char *file_frame;              /**< Name of the file with frames.                    */
   char *file_partitura;          /**< Name of the file with partiture.                 */
   char *file_kmax;               /**< Name of the file with vector kmax data.          */
   char *file_kmin;               /**< Name of the file with vector kmin data.          */
   char *fileStates_Time_e;       /**< Name of the file with vector States_Time_e data. */
   char *fileStates_Time_i;       /**< Name of the file with vector States_Time_i data. */
   char *fileStates_seq;          /**< Name of the file with vector States_seq data.    */
   char *fileStates_corr;         /**< Name of the file with vector corr_states data.   */
};
typedef struct DTWconst  DTWconst;
typedef struct DTWfiles  DTWfiles;

/**
 *  \struct WAVHeader
 *  \brief Struct for WAVE file header.<BR>
 *  WAV files definitions. Note that we only work with 1 channel, 2 bytes per sample
 *  FMT length 16, PCM format, standard header length (44 bytes). See Read_WAVHeader
 *  function within FileFunctions.c file and struct WAVHeader within this file
*/
struct WAVHeader
{
    char riff[4];                  /**< String "RIFF"                                  */
    unsigned int size;             /**< overall size of file in bytes                  */
    char wave[4];                  /**< String "WAVE"                                  */
    char  fmt[4];                  /**< String "fmt" with trailing null char           */
    unsigned int fmt_length;       /**< length of the format data                      */
    unsigned int format_type;      /**< format type. 1 PCM, 3 IEEE float, ...          */
    unsigned int channels;         /**< number of channels                             */
    unsigned int sample_rate;      /**< sampling rate (samples per second or  Hertz)   */
    unsigned int byte_rate;        /**< (sample_rate * channels * bits_per_sample)/8   */
    unsigned int block_align;      /**< Number bytes for sample including all channels */
    unsigned int bits_per_sample;  /**< bits per sample, 8 8bits, 16 16bits, ...       */
    char data_header[4];           /**< String "DATA" or "FLLR" string                 */
    unsigned int data_size;        /**< Size of the data section                       */
    long num_samples;              /**< num_samples = data_size / block_align          */
    unsigned int bytes_per_sample; /**< bytes_per_sample = bits_per_sample*channels/8  */
};
typedef struct WAVHeader WAVHeader;


/**
 *  \struct STempo Gaussian
 *  \brief Struct for Compute tempos.<BR>
*/
struct STempo
{
    int MidiFrame;
    int RealFrame;
    int NextFrame;
    int PrevState;
    MyType SynthSpeed;
    MyType SoloSpeed;
    MyType SynthTime;
};
typedef struct STempo STempo;

/**
 *  \struct STempoRL Regresion Lineal
 *  \brief Struct for storing .<BR>
*/
struct STempoRL
{
    int NextFrame;
    int PrevState;
    int numap;
    int matched;
    int AudioTimeAP[NUMAPMAX];
    int ScoreTimeAP[NUMAPMAX];
    MyType SynthSpeed;
    MyType SynthTime;
    MyType SoloSpeed;
};
typedef struct STempoRL STempoRL;


#define N_MAXVEL  8
#define M_SQRT1_2 0.707106781186547524401
//MOD 2018-12-22
//#define WCOSTE    8
#define ANTCF     2

#define CDFGAUSS(x,m,s) (0.5 * erfc(((m)-(x)) * M_SQRT1_2 / (s)))
#define CALCVEL(e,t)    ((MyType)(e) / (MyType)(t))

#define NSEQTEST 418
#define JTEST 431

/**
 *  \struct SVelStates
 *  \brief Struct for storing info about velocity of interpretation.<BR>
*/
struct SVelStates
{
    int *states_dur;     /**< Duration of each state                          */
    int *onset_time;     /**< Last onset time for each DTW cell               */
    MyType *avg_vel;     /**< Mean of velocities for each DTW cell            */
    MyType *std_vel;     /**< Std deviation of velocities for each DTW cell   */
    MyType *velocities;  /**< Last velocities for each DTW cell               */
    int *nvel;           /**< Number of velocities for each DTW cell          */
};
typedef struct SVelStates SVelStates;

#endif
