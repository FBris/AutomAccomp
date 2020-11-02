 /**************************************************************************
 *   Copyright (C) 2017 by "Information Retrieval and Parallel Computing" *
 *   group (University of Oviedo, Spain), "Interdisciplinary Computation  *
 *   and Communication" group (Polytechnic University of Valencia, Spain) *
 *   and "Signal Processing and Telecommunication Systems Research" group *
 *   (University of Jaen, Spain)                                          *
 *   Contact: remaspack@gmail.com                                         *
 *                                                                        *
 *   This program is free software; you can redistribute it and/or modify *
 *   it under the terms of the GNU      Public License as published by *
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
 *  \file    ServerCPU.c
 *  \brief   Server version of ReMAS for CPU, both x86_64 and ARM.
 *  \author  Information Retrieval and Parallel Computing Group, Univºersity of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/

/* This include the first. Only needed when function for waiting some milliseconds is used */
#include "TimeFunctions.h"

#include "CPUFunctions.h"
#include "FileFunctions.h"
#include "SoundFunctions.h"
#include "TempoFunctions.h"
#include "NetFunctions.h"

#ifdef CAUDIO
  #include "MacOSSoundFunctions.h"
  #pragma mark main function
#endif
/**
 *  \fn    int main(int argc , char *argv[])
 *  \brief main is the entry point to ReMAS, classical C main program.
 *  \param argc: (in) As is usual in programs C
 *  \param argv: (in) As is usual in programs C
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int main(int argc , char *argv[])
{
   /* Using only by DTW */
   int
     pos_min,
     pos_sil,
     DTWWhere, DTWSize,
     DTWSizePlusPad;

   MyType
     *pD = NULL,
     Costs[T_COSTS];

   /* Using only by FFT */
   MyFFTCPUType
     plan;

   MyType
     *X_fft  =NULL,
     *Out_fft=NULL,
     *Mod_fft=NULL;

   int
     kmax_fft[N_MIDI],
     kmin_fft[N_MIDI];

   /* Using only by HMM (silence detection) */
   MyType
     prob_silen = (MyType)0.75,
     prob_audio = (MyType)0.25;

   bool
     Silence=true;

   /* Using by other preprocessing steps */
   MyType
     *norms     = NULL,
     *s_fk      = NULL,
     *v_SxD     = NULL,
     *v_cfreq   = NULL,
     *v_hanning = NULL,
     *v_dxState = NULL,
     *tauxi     = NULL,
     *ts_fk     = NULL,
     BETA=(MyType)1.5,
     WCOSTE = (MyType) 0.0;

   int
     *states_time_i=NULL,
     *states_time_e=NULL,
     *states_seq   =NULL,
     *states_corr  =NULL,
     *I_SxD        =NULL;

   short
     *frame=NULL;

   DTWconst  Param;
   DTWfiles  NameFiles;
   WAVHeader WavHeader;

   /* For Linux when audio comes from microphone input device */
   #ifdef ALSA
     snd_pcm_t         *SoundHandle=NULL;
     snd_pcm_uframes_t  SoundBufferSize;
   #endif

   /* For MacOS when audio comes from microphone input device */
   #ifdef CAUDIO
     RecorderStruct recorder = {0};
     AudioQueueRef queue = {0};
   #endif

   /* Using by OSC for messages. Limited to MaxOSC (see defines.h) clients */
   #ifdef OSC
     lo_address DirOSC[MaxOSC];
   #endif

   /* For TEMPO */
   //STempo TEMPO={.MidiFrame=0, .RealFrame=0, .NextFrame=0, .PrevState=0, .SynthSpeed=1.0, .SoloSpeed=.0, .SynthTime=.0};
   STempoRL TEMPORL={.NextFrame=0,   .PrevState=0,   .AudioTimeAP[0]=0, .ScoreTimeAP[0]=0, .SynthSpeed=1.0,
                     .SoloSpeed=1.0, .SynthTime=0.0, .numap=1,          .matched=1};
   int *preload=NULL;
   SVelStates SVT;

   /* For Docker and AppImage usage */
   bool UseOSC=false, UseMic=false;

   /* For time under OpenMP */
   #ifdef OMP
     double time=0.0;
   #endif

   /* General & others variables */
   int i, j, NumTramas=0;
   int NumSilence = -1;
   int stateprev = 0;
   FILE *fp = NULL,
        *fp_pth = NULL,
        *fp_sxd = NULL,
        *fp_nor = NULL,
        *fp_aln = NULL,
        *fp_var = NULL;

   /* Reading global paramentes */
   switch(argc) {
      case 3: /* Regular use */
        #ifdef SIMPLE
          BETA=strtof(argv[1], NULL);
        #else
          BETA=strtod(argv[1], NULL);
        #endif
        #ifdef OSC
          UseOSC=true;
        #endif
        #if defined(ALSA) || defined(CAUDIO)
          UseMic=true;
        #endif
        break;
      case 6: /* For Docker or AppImagen */
        #ifdef SIMPLE
          BETA=strtof(argv[1], NULL);
        #else
          BETA=strtod(argv[1], NULL);
        #endif
        UseOSC=atoi(argv[3]);
        UseMic=atoi(argv[4]);
        // NEW 2018-12-22:
        WCOSTE=atof(argv[5]);

        break;
      default:
        printf("\nGeneral usage: %s <BETA> <configuration file>\n", argv[0]);
        printf("   Example: %s 1.5 parametes.dat\n\n", argv[0]);
        printf("Docker  usage: %s <BETA> <configuration file> <OSC yes|no [1|0]> <Microphone yes|no [1|0]>\n", argv[0]);
        printf("   Example: %s 1.5 parametes.dat 1 1\n\n", argv[0]);
        return -1;
   }

   /* Reading general information and file names */
   CHECKERR(ReadParameters(&Param, &NameFiles, argv[2]));

   /* Allocating memory and reading some structures */
   CHECKERR(AllocDataCPU(&v_hanning, &states_time_i, &states_time_e, &states_seq, &states_corr, &I_SxD, &DTWSize, TAMTRAMA, Param.N_STATES, NameFiles));

   /* Allocating memory for and setting some DTW constants */
   DTWSizePlusPad =(DTWSize +  N_COSTS) * (N_COSTS + 1);
   CHECKERR(AllocDTWCPU(&pD, &v_SxD, DTWSize, DTWSizePlusPad, 1));

   /* Allocating memory for s_fk and auxiliar structures when Beta !=0.0 and !=1.0 */
   CHECKERR(AllocS_fkCPU(&s_fk, &tauxi, &ts_fk, BETA, N_MIDI_PAD, Param.N_BASES, NameFiles));

   /* Allocating memory for FFT memory and reading data */
   CHECKERR(AllocFFTCPU(&plan, &X_fft, &Out_fft, &Mod_fft, kmin_fft, kmax_fft, N_FFT, NameFiles));

   /* Allocating memory for the rest of general structures */
   CHECKERR(AllocAuxiCPU(&norms, &frame, &v_cfreq, &v_dxState, Param.N_BASES, TAMTRAMA, N_MIDI));

   /* Initializing vector of costs */
   Costs[0] = 1.0; // Distance to (1,1)
   Costs[1] = 1.0; // Distance to (1,2)
   Costs[2] = 1.0; // Distance to (1,3)
   Costs[3] = 1.0; // Distance to (1,4)
   Costs[4] = 2.0; // Distance to (2,1)
   Costs[5] = 3.0; // Distance to (3,1)
   Costs[6] = 4.0; // Distance to (4,1)

   /* Ouput files */
   CHECKNULL(fp_var  = fopen("var.txt", "w"));
   CHECKNULL(fp_nor  = fopen("nor.txt", "w"));
   CHECKNULL(fp_pth  = fopen("pth.txt", "w"));
   CHECKNULL(fp_sxd  = fopen("sxd.bin", "w"));
   CHECKNULL(fp_aln  = fopen("alignm.txt", "w"));

   /* State of each score frame */
   CHECKNULL( preload = (int *) calloc(DTWSize, sizeof(int)) );
   j=0;
   for (i=0; i<DTWSize; i++)
   {
     if (i > states_time_e[j]) j++;
     preload[i] = j;
   }
   CHECKERR(AllocSVelCPU(states_time_i, states_time_e, Param.N_STATES, DTWSize, DTWSizePlusPad, &SVT));

   fclose(fp_var);
   fclose(fp_nor);


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
     #ifdef CAUDIO
       if (UseMic)
       {
         ConfigureAndAllocAudioQueues(&recorder, &queue);
         recorder.running = TRUE;
         CheckError(AudioQueueStart(queue, NULL), "AudioQueueStart failed");

         #ifdef DUMP
           CHECKNULL(fp=fopen("FramesMicRecorded.pcm", "wb"));
         #endif
         NumTramas = (Param.Time_MIC * AQRate) / TAMMUESTRA;
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
   #endif

   /* Compute s_fk norms */
   ComputeNorms(norms, ts_fk, s_fk, N_MIDI, Param.N_BASES, BETA);

   /* Fills the buffer */
   #ifdef ALSA
     if (UseMic)
       CHECKERR(ReadAlsaCPU1st(frame, SoundHandle, fp));
     else
       CHECKERR(ReadWavCPU1st(frame, fp));
   #else
     #ifdef CAUDIO
       if (UseMic)
         CHECKERR(ReadAudioQueue1st(frame, &recorder, fp));
       else
         CHECKERR(ReadWavCPU1st(frame, fp));
     #else
       CHECKERR(ReadWavCPU1st(frame, fp));
     #endif
   #endif

   #ifdef TALK
     printf("Listening ...\n");
   #endif

   /* Procedure for silence/white noise detection */
   while (Silence)
   {
     #ifdef ALSA
       if (UseMic)
         CHECKERR(ReadAlsaCPU(frame, SoundHandle, fp));
       else
         CHECKERR(ReadWavCPU(frame, fp));
     #else
       #ifdef CAUDIO
         if (UseMic)
           CHECKERR(ReadAudioQueue(frame, &recorder, fp));
         else
           CHECKERR(ReadWavCPU(frame, fp));
       #else
         CHECKERR(ReadWavCPU(frame, fp));
       #endif
     #endif
     NumTramas--;
     NumSilence++;

     ApplyWindow (X_fft, frame, v_hanning, TAMTRAMA, N_FFT);
     FFT         (v_cfreq, kmin_fft, kmax_fft, X_fft, Mod_fft, Out_fft, plan, N_FFT, N_MIDI);
     BetaNormBien(v_cfreq, N_MIDI, BETA);
     ComputeDist (v_cfreq, v_dxState, tauxi, norms, s_fk, ts_fk, BETA, Param.N_BASES, N_MIDI);
     pos_sil = SeqIdamin(Param.N_BASES-1, v_dxState+1) + 1;
     Silence = DetectSilence((v_dxState[pos_sil]-0.90*v_dxState[0]), &prob_silen, &prob_audio);
     //Silence = DetectSilence((v_dxState[1]-v_dxState[0]), &prob_silen, &prob_audio);
     //Silence = 0;
   }
   printf("Hay %d tramas de silencio inicial.\n", NumSilence);

   /* Start OSC */
   #ifdef OSC
     if (UseOSC) for (i=0; i<Param.NCliOSC; i++) { CHECKERR(SendPlay(DirOSC[i])); CHECKERR(SendTempo(DirOSC[i], 110)); }
   #endif

   #ifdef OMP
     time=omp_get_wtime();
   #endif

   /* start the system */
   j = 0;
   for(i=1; i<=NumTramas; i++)
   {
      ApplyWindow (X_fft,     frame,     v_hanning, TAMTRAMA, N_FFT);
      FFT         (v_cfreq,   kmin_fft,  kmax_fft,  X_fft,    Mod_fft, Out_fft, plan, N_FFT,         N_MIDI);
      BetaNormBien(v_cfreq, N_MIDI,    2);  // SILENCIO
      ComputeDist (v_cfreq,   v_dxState, tauxi,     norms,    s_fk,    ts_fk,   BETA, Param.N_BASES, N_MIDI);

      // SILENCIO
      pos_sil = SeqIdamin(Param.N_BASES-1, v_dxState+1) + 1;
      Silence = DetectSilence((v_dxState[pos_sil]-v_dxState[0]), &prob_silen, &prob_audio);

      // DTW
      if (Silence==0)
      {
         /* code */
         j++;
         ApplyDist  (v_dxState, v_SxD,     I_SxD,     DTWSize,  Param.ALPHA);
         DTWWhere = (j % TBLOCK) * (N_COSTS + DTWSize) + N_COSTS;
         //pos_min  = DTWProcVar(v_SxD, Costs, pD, i, DTWWhere, DTWSize);
         //pos_min=DTWProc(v_SxD, Costs, pD, i, DTWWhere, DTWSize);
         pos_min = DTWProcMed(v_SxD, Costs, pD, j, DTWWhere, DTWSize, states_corr, preload, &SVT, WCOSTE);

         fprintf(fp_pth, "%d\t%d\n", i+NumSilence, pos_min+1);
         //fwrite (v_SxD,  sizeof(MyType), DTWSize, fp_sxd);
         WriteAlignment(i+NumSilence, pos_min, &stateprev, preload, fp_aln);
      }

      // Read new data
      #ifdef ALSA
        if (UseMic)
          CHECKERR(ReadAlsaCPU(frame, SoundHandle, fp));
        else
          CHECKERR(ReadWavCPU(frame, fp));
      #else
        #ifdef CAUDIO
          if (UseMic)
            CHECKERR(ReadAudioQueue(frame, &recorder, fp));
          else
           CHECKERR(ReadWavCPU(frame, fp));
        #else
          CHECKERR(ReadWavCPU(frame, fp));
        #endif
      #endif
   }

   #ifdef OMP
     time=omp_get_wtime() - time;
     printf("%f sec.\n", time);
   #endif

   /* Leave MS on tempo 1, stop and frees*/
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
     #ifdef CAUDIO
       if (UseMic)
       {
         #ifdef DUMP
           fclose(fp);
         #endif
         recorder.running = FALSE;
         CheckError(AudioQueueStop(queue, TRUE), "AudioQueueStop failed");
         AudioQueueDispose(queue, TRUE);
       } else { fclose(fp); }
     #else
       fclose(fp);
     #endif
   #endif

   fclose(fp_sxd);
   fclose(fp_pth);
   fclose(fp_aln);
   /* frees in general. Is it necessary ? Not really but it is our habit */
   if (!(BETA>=(MyType)0.0 && BETA<=(MyType)0.0) && !(BETA>=(MyType)1.0 && BETA<=(MyType)1.0)) {
     free(tauxi);
     free(ts_fk);
   }
   free(I_SxD);
   free(Mod_fft);
   free(norms);
   free(pD);
   free(s_fk);
   free(preload);
   free(states_corr);
   free(states_seq);
   free(states_time_i);
   free(states_time_e);
   free(frame);
   free(v_cfreq);
   free(v_dxState);
   free(v_hanning);
   free(v_SxD);
   free(X_fft);
   #ifdef SIMPLE
     fftwf_free(Out_fft);
     fftwf_destroy_plan(plan);
     #ifdef PARFFTW
       fftwf_cleanup_threads();
     #endif
   #else
     fftw_free(Out_fft);
     fftw_destroy_plan(plan);
     #ifdef PARFFTW
       fftw_cleanup_threads();
     #endif
   #endif

   return OK;
}
