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
 *  \file    CPUFunctions.c
 *  \brief   File with code of ReMAS functions for CPU, both x86_64 and ARM.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#include "CPUFunctions.h"
#include "FileFunctions.h"


/**
 *  \fn    void ApplyWindow(MyType * __restrict X_fft, const short *frame, const MyType *v_hanning, const int framesize, const int xfftsize)
 *  \brief ApplyWindow applies hanning window to the current frame and store the result en vector X_fft
 *  \param X_fft:     (out) Vector X_fft to update
 *  \param frame:      (in) Vector with current frame
 *  \param v_hanning:  (in) Hanning vector
 *  \param framesize:  (in) Size of the frame
 *  \param xfftsize:   (in) Size of the vector X_fft
 *  \return: Nothing, it is void
*/
void ApplyWindow(MyType * __restrict X_fft, const short *frame, const MyType *v_hanning,
                 const int framesize, const int xfftsize)
{
   int i;

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(i=0; i<framesize; i++) { X_fft[i]=(MyType)frame[i] * Scaling * v_hanning[i]; }

   memset(&X_fft[framesize], 0, sizeof(MyType)*(xfftsize - framesize));
}


/**
 *  \fn    void ComputeNorms(MyType *norms, MyType *ts_fk, const MyType *s_fk, const int nmidi, const int nbases, const MyType beta)
 *  \brief ComputeNorms fills norms vector
 *  \param norms: (out) Vector with the norms
 *  \param ts_fk:  (in) Auxiliar vector
 *  \param s_fk:   (in) Vector s_fk
 *  \param nmidi:  (in) Number of midi notes
 *  \param nbases: (in) Number of bases/combinations
 *  \param beta:   (in) BETA value
 *  \return: Nothing, it is void
*/
void ComputeNorms(MyType *norms, MyType *ts_fk, const MyType *s_fk, const int nmidi,
                  const int nbases, const MyType beta)
{
   int i;

   if (beta>=(MyType)0.0 && beta<=(MyType)0.0)
      for(i=0; i<nbases; i++) { norms[i]=(MyType)nmidi; }
   else if (beta>=(MyType)1.0 && beta<=(MyType)1.0)
   {
      #ifdef OMP
        #pragma omp parallel for
      #endif
      for(i=0; i<nbases; i++)
      {
         int k;
         #ifndef CBLAS
            int j;
            MyType data;
         #endif

         k=i*N_MIDI_PAD;

         #ifdef CBLAS
            #ifdef SIMPLE
               norms[i]=cblas_sasum(nmidi, &s_fk[k], 1);
            #else
               norms[i]=cblas_dasum(nmidi, &s_fk[k], 1);
            #endif
         #else
            data=(MyType)0.0;
            for(j=0; j<nmidi; j++)
               data += s_fk[k+j];
            norms[i] = data;
         #endif
      }
   }
   else
   {
      #ifdef CBLAS
        for(i=0; i<nbases; i++)
           #ifdef SIMPLE
             norms[i]=cblas_sdot(nmidi, &ts_fk[i*N_MIDI_PAD], 1, &s_fk[i*N_MIDI_PAD], 1);
           #else
             norms[i]=cblas_ddot(nmidi, &ts_fk[i*N_MIDI_PAD], 1, &s_fk[i*N_MIDI_PAD], 1);
           #endif
      #else
        #ifdef OMP
          #pragma omp parallel for
        #endif
        for(i=0; i<nbases; i++)
        {
           int    j, k;
           MyType data;

           k=i*N_MIDI_PAD;

           data=(MyType)0.0;

           for(j=0; j<nmidi; j++)
              data += ts_fk[k+j]*s_fk[k+j];
           norms[i]=data;
        }
      #endif
   }
}


/**
 *  \fn    int AllocAuxiCPU(MyType **norms, short **frame, MyType **v_cfreq, MyType **v_dxState, const int nbases, const int tamframe, const int nmidi)
 *  \brief AllocAuxiCPU Memory reservation for norms, frame, v_cfreq and v_dxState vectors
 *  \param norms:     (out) Norms vector
 *  \param frame:     (out) Vector for frames
 *  \param v_cfreq:   (out) v_cfreq vector
 *  \param v_dxState: (out) v_dxState vector
 *  \param nbases:     (in) Number of bases/combinations, sizeof norms and v_dxState
 *  \param tamframe:   (in) Size of frames in samples
 *  \param nmidi:      (in) Number of midi notes, sizeof v_cfreq
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocAuxiCPU(MyType **norms, short **frame, MyType **v_cfreq, MyType **v_dxState, const int nbases,
                 const int tamframe, const int nmidi)
{
   CHECKNULL((*norms)    =(MyType *)calloc(nbases,   sizeof(MyType)));
   CHECKNULL((*v_dxState)=(MyType *)calloc(nbases,   sizeof(MyType)));
   CHECKNULL((*frame)    =(short *) calloc(tamframe, sizeof(short)));
   CHECKNULL((*v_cfreq)  =(MyType *)calloc(nmidi,    sizeof(MyType)));

   return OK;
}


/**
 *  \fn    int AllocFFTCPU(MyFFTCPUType *plan, MyType **X_fft, MyType **Out_fft, MyType **Mod_fft, int *kmin_fft, int *kmax_fft, const int nfft, DTWfiles NameFiles)
 *  \brief AllocFFTCPU Allocates memory for FFT vector and reads some fft information from files
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
int AllocFFTCPU(MyFFTCPUType *plan, MyType **X_fft, MyType **Out_fft, MyType **Mod_fft, int *kmin_fft,
                int *kmax_fft, const int nfft, DTWfiles NameFiles)
{

   CHECKNULL((*X_fft)  =(MyType *)calloc(2*nfft+1, sizeof(MyType)));
   CHECKNULL((*Mod_fft)=(MyType *)calloc(nfft,     sizeof(MyType)));

   #ifdef SIMPLE
      CHECKNULL((*Out_fft)=(MyType *)fftwf_malloc(sizeof(MyType)*nfft));

      #ifdef PARFFTW
         fftwf_init_threads();
         #ifdef OMP
            fftwf_plan_with_nthreads(omp_get_max_threads());
         #else
            fftwf_plan_with_nthreads(sysconf(_SC_NPROCESSORS_CONF));
         #endif
      #endif

      CHECKNULL((*plan)=fftwf_plan_r2r_1d(nfft, (*X_fft), (*Out_fft), FFTW_R2HC, FFTW_MEASURE));
   #else
      CHECKNULL((*Out_fft)=(MyType *)fftw_malloc(sizeof(MyType)*nfft));

      #ifdef PARFFTW
         fftw_init_threads();
         #ifdef OMP
            fftw_plan_with_nthreads(omp_get_max_threads());
         #else
            fftw_plan_with_nthreads(sysconf(_SC_NPROCESSORS_CONF));
         #endif
      #endif

      CHECKNULL((*plan)= fftw_plan_r2r_1d(nfft, (*X_fft), (*Out_fft), FFTW_R2HC, FFTW_MEASURE));
   #endif

   CHECKERR(ReadVectorInt64(kmax_fft, N_MIDI, NameFiles.file_kmax));
   CHECKERR(ReadVectorInt64(kmin_fft, N_MIDI, NameFiles.file_kmin));

   return OK;
}


/**
 *  \fn    int AllocS_fkCPU(MyType **s_fk, MyType **tauxi, MyType **ts_fk, const MyType BETA, const int nmidi, const int nbases, DTWfiles NameFiles)
 *  \brief AllocS_fkCPU Allocates memory for S_fk vector, read its data from file and initializes other auxiliar vectors
 *  \param s_fk:     (out) s_fk vector
 *  \param tauxi:    (out) Auxiliar vector tauxi
 *  \param ts_fk:    (out) Auxiliar vector ts_fk
 *  \param BETA:      (in) BETA value
 *  \param nmidi:     (in) Number of midi notes
 *  \param nbases:    (in) Number of bases/combinations
 *  \param NameFiles: (in) Struct with the file names
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocS_fkCPU(MyType **s_fk, MyType **tauxi, MyType **ts_fk, const MyType BETA, const int nmidi,
                 const int nbases, DTWfiles NameFiles)
{
   int i;

   CHECKNULL((*s_fk)=(MyType *)calloc(nmidi*nbases, sizeof(MyType)));

   CHECKERR(ReadS_fk((*s_fk), nbases, NameFiles.file_partitura));

   if (!(BETA>=(MyType)0.0 && BETA<=(MyType)0.0) && !(BETA>=(MyType)1.0 && BETA<=(MyType)1.0))
   {
      CHECKNULL((*tauxi)=(MyType *)calloc(nmidi,        sizeof(MyType)));
      CHECKNULL((*ts_fk)=(MyType *)calloc(nmidi*nbases, sizeof(MyType)));

      #ifdef OMP
        #pragma omp parallel for
      #endif
      for (i=0; i<nmidi*nbases; i++)
         #ifdef SIMPLE
            (*ts_fk)[i]=powf((*s_fk)[i], BETA-1.0f);
         #else
            (*ts_fk)[i]= pow((*s_fk)[i], BETA-1.0);
         #endif
   }

   return OK;
}


/**
 *  \fn    int AllocDTWCPU(MyType **pV, MyType **v_SxD, const int DTWSize, const int DTWSizePlusPad, const int startin)
 *  \brief AllocDTWCPU Allocates memory for DTW vectors and initializes them
 *  \param pV:            (out) DTW pV vector
 *  \param v_SxD:         (out) v_SxD vector
 *  \param DTWSize:        (in) Size of DTW vectors
 *  \param DTWSizePlusPad: (in) Size of DTW vectors plus padding
 *  \param startin         (in) Where we start to store data within circular-buffer
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocDTWCPU(MyType **pV, MyType **v_SxD, const int DTWSize, const int DTWSizePlusPad, const int startin)
{
   int i;

   CHECKNULL((*pV)   =(MyType *)malloc(sizeof(MyType)*DTWSizePlusPad));
   CHECKNULL((*v_SxD)=(MyType *)calloc(DTWSize, sizeof(MyType)));

   for (i=0; i<DTWSizePlusPad; i++)
      #ifdef SIMPLE
        (*pV)[i]=FLT_MAX;
      #else
        (*pV)[i]=DBL_MAX;
      #endif

   /* Where we start to store data within circular-buffer */
   i = startin % TBLOCK;
   if (i==0)
      (*pV)[DTWSizePlusPad-DTWSize] = 0.0;
   else
      (*pV)[i*(DTWSize + N_COSTS) - DTWSize] = 0.0;

   return OK;
}


/**
 *  \fn    int AllocDataCPU(MyType **v_hanning, int **states_time_i, int **states_time_e, int **states_seq, int **states_corr, int **I_SxD, int *DTWSize, const int tamframe, const int nstates, DTWfiles NameFiles)
 *  \brief AllocDataCPU creates and initializes some structures reading info from files
 *  \param v_hanning:     (out) v_hanning vector
 *  \param states_time_i: (out) states_time_i vector, contains the start-time of each state in frames
 *  \param states_time_e: (out) states_time_e vector, contains the end-time of each state in frames
 *  \param states_seq:    (out) states_seq vector, contains the base/combination that is performed in each state
 *  \param states_corr:   (out) states_corr vector
 *  \param I_SxD:         (out) I_SxD vector
 *  \param DTWSize:        (in) Size of DTW vectors
 *  \param tamframe:       (in) Size of frames in samples
 *  \param nstates:        (in) Number of states
 *  \param NameFiles:      (in) Struct with the file names
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int AllocDataCPU(MyType **v_hanning, int **states_time_i, int **states_time_e, int **states_seq, int **states_corr,
                 int **I_SxD, int *DTWSize, const int tamframe, const int nstates, DTWfiles NameFiles)
{
   int
     i, j, pos;

   CHECKNULL((*v_hanning)    =(MyType *)calloc(tamframe, sizeof(MyType)));
   CHECKNULL((*states_time_i)=   (int *)calloc(nstates,  sizeof(int)));
   CHECKNULL((*states_time_e)=   (int *)calloc(nstates,  sizeof(int)));
   CHECKNULL((*states_seq)   =   (int *)calloc(nstates,  sizeof(int)));
   CHECKNULL((*states_corr)  =   (int *)calloc(nstates,  sizeof(int)));

   CHECKERR(      ReadVector((*v_hanning),    tamframe, NameFiles.file_hanning));
   CHECKERR(ReadVectorInt64((*states_seq),    nstates,  NameFiles.fileStates_seq));
   CHECKERR(ReadVectorInt64((*states_time_i), nstates,  NameFiles.fileStates_Time_i));
   CHECKERR(ReadVectorInt64((*states_time_e), nstates,  NameFiles.fileStates_Time_e));
   CHECKERR(ReadVectorInt64((*states_corr),   nstates,  NameFiles.fileStates_corr));

   (*DTWSize) = (*states_time_e)[nstates - 1] + 1;

   CHECKNULL((*I_SxD)=(int *)calloc((*DTWSize), sizeof(int)));

   pos=0;
   for (i=0; i<nstates; i++)
      for (j=(*states_time_i)[i]; j<=(*states_time_e)[i]; j++)
      {
         (*I_SxD)[pos]=(*states_seq)[i];
         pos++;
      }
   return OK;
}


int AllocSVelCPU(const int *states_time_i, const int *states_time_e, const int nstates,
                 const int DTWSize, const int DTWSizePlusPad, SVelStates *SVT)
{
    int i;
    int *states_dur;
    int *onset_time;
    MyType *velocities;
    MyType *avg_vel;
    MyType *std_vel;
    int *nvel;

    /* Duration of each state */
    CHECKNULL( states_dur = (int *) malloc(nstates * sizeof(int)) );
    for (i=0; i<nstates; i++)
       states_dur[i] = states_time_e[i] - states_time_i[i] + 1;
    SVT->states_dur = states_dur;

    /* Last onset time for each DTW cell */
    CHECKNULL( onset_time = (int *) malloc(DTWSizePlusPad * sizeof(int)) );
    SVT->onset_time = onset_time;

    /* Number of velocities for each DTW cell */
    CHECKNULL( nvel = (int *) calloc(DTWSizePlusPad, sizeof(int)) );
    SVT->nvel = nvel;

    /* Last velocities for each DTW cell */
    CHECKNULL( velocities = (MyType *) malloc(DTWSizePlusPad * N_MAXVEL * sizeof(MyType)) );
    SVT->velocities = velocities;

    /* Mean of velocities for each DTW cell */
    CHECKNULL( avg_vel = (MyType *) malloc(DTWSizePlusPad * sizeof(MyType)) );
    SVT->avg_vel = avg_vel;

    /* Std deviation of velocities for each DTW cell */
    CHECKNULL( std_vel = (MyType *) malloc(DTWSizePlusPad * sizeof(MyType)) );
    SVT->std_vel = std_vel;

    return OK;
}


/**
 *  \fn    void FFT(MyType *v_cfreq, const int *kmin, const int *kmax, MyType *X_fft, MyType *Mod_fft, MyType *Out_fft, MyFFTCPUType plan, const int nfft, const int nmidi)
 *  \brief FFT computes FFT and updates v_cfreq vector
 *  \param v_cfreq:   (out) v_cfreq vector
 *  \param kmin:    (input) kmin vector
 *  \param kmax:    (input) kmax vector
 *  \param X_fft:   (inout) X_fft vector
 *  \param Mod_fft: (inout) Mod_fft vector
 *  \param Out_fft: (inout) Out_fft vector
 *  \param plan:      (out) FFT scheduler
 *  \param nfft:       (in) As is to be, Mod_fft and Out_fft size
 *  \param nmidi:      (in) Number of midi notes
 *  \return: Nothing, it is void
*/
void FFT(MyType *v_cfreq, const int *kmin, const int *kmax, MyType *X_fft, MyType *Mod_fft,
         MyType *Out_fft, MyFFTCPUType plan, const int nfft, const int nmidi)
{
   int i;

   #ifdef SIMPLE
      fftwf_execute(plan);
   #else
      fftw_execute(plan);
   #endif

   Mod_fft[0]     = Out_fft[0]     * Out_fft[0];
   Mod_fft[nfft/2]= Out_fft[nfft/2]* Out_fft[nfft/2];

   #ifdef OMP
    #pragma omp parallel for
   #endif
   for(i=1; i<nfft/2; i++)
      Mod_fft[i]=Out_fft[i]*Out_fft[i] + Out_fft[nfft-i]*Out_fft[nfft-i];

   #ifdef CBLAS
     for(i=0;i<nmidi;i++)
        #ifdef SIMPLE
          v_cfreq[i]=sqrtf(cblas_sasum(kmax[i]-kmin[i]+1, &Mod_fft[kmin[i]], 1));
        #else
          v_cfreq[i]= sqrt(cblas_dasum(kmax[i]-kmin[i]+1, &Mod_fft[kmin[i]], 1));
        #endif
   #else
     #ifdef OMP
       #pragma omp parallel for
     #endif
     for(i=0;i<nmidi;i++)
     {
        int j;
        MyType value;

        value=0.0;
        for(j=kmin[i];j<=kmax[i];j++) value+=Mod_fft[j];

        #ifdef SIMPLE
          v_cfreq[i]=sqrtf(value);
        #else
          v_cfreq[i]= sqrt(value);
        #endif
     }
   #endif
}


#ifdef OMP
/**
 *  \fn    int ParIdamin(const int N, const MyType *v)
 *  \brief ParIdamin returns the pos of the 1st minimum in a vector. It is a parallel function.
 *  \param N: (in) Vector size
 *  \param v: (in) The vector
 *  \return: The pos of "a" minimum
*/
int ParIdamin(const int N, const MyType *v)
{
  int i;

  #ifdef OMP4
     PosMin data ={DBL_MAX, 0};

     #pragma omp parallel for reduction(MIN: data)
     for (i=0; i<N; i++)
        if (v[i] < data.val) { data.val=v[i]; data.pos=i; }

     return data.pos;
  #else
     int    pos_min = 0;
     MyType val_min = v[0];

     #pragma omp parallel
     {
       int pos=1;
       #ifdef SIMPLE
         MyType val = FLT_MAX;
       #else
         MyType val = DBL_MAX;
       #endif

       #pragma omp for nowait
       for(i=1; i<N; i++)
         if (v[i]<val) { val=v[i]; pos=i; }

       #pragma omp critical (ZonaCritica1)
       {
         if (val < val_min) { val_min=val; pos_min=pos; }
         else if ((val==val_min) && (pos < pos_min)) pos_min=pos;
       }
     }
     return pos_min;
  #endif
}
#endif


/**
 *  \fn    int SeqIdamin(const int N, const MyType *v)
 *  \brief SeqIdamin returns the pos of the minimum in a vector.
 *  \brief When minimum's number of occurrences > 1 => returns the position of the 1st occurrence
 *  \param N: (in) Vector size
 *  \param v: (in) The vector
 *  \return: The pos of the minimum
*/
int SeqIdamin(const int N, const MyType *v)
{
   int i, pos=0;
   MyType val=v[0];

   for(i=1; i<N; i++)
     if (v[i]<val) { pos=i; val=v[i]; }

   return pos;
}


/**
 *  \fn    int DTWProc(const MyType *Sequence, const MyType *Costs, MyType * __restrict pD, const int NSeq, const int Where, const int NST)
 *  \brief DTWProc performs the Online-DTW process for the current frame
 *  \param Sequence: (in) Currente frame
 *  \param Costs:    (in) Vector with DTW path-costs
 *  \param pD:      (out) DTW pD vector
 *  \param NSeq:     (in) One referenced position within DTW vectors
 *  \param Where:    (in) Position within DTW vectors
 *  \param NST:      (in) Number of states
 *  \return: Position of the minimun
*/
int DTWProc(const MyType *Sequence, const MyType *Costs, MyType * __restrict pD, const int NSeq, const int Where, const int NST)
{
   int NSTplusNC, Ref_Pos, j;

   /* Some auxilar varibles */
   NSTplusNC = N_COSTS + NST;
   Ref_Pos   = ((NSeq + N_COSTS) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(j=0; j<NST; j++)
   {
      MyType d, d2;
      int    k, Pos;

      #ifdef SIMPLE
         d=FLT_MAX;
      #else
         d=DBL_MAX;
      #endif

      Pos=Ref_Pos + j;

      for(k=0; k<N_COSTS; k++)
      {
         d2 = Sequence[j]*Costs[k]+pD[Pos-k];
         if (d2 < d) d=d2;
      }

      for (k=N_COSTS; k<T_COSTS; k++)
      {
         Pos=((NSeq + (T_COSTS-k)) % TBLOCK) * NSTplusNC + N_COSTS + j - 1;

         d2 = Sequence[j]*Costs[k]+pD[Pos];

         if (d2 < d) d=d2;
      }

      pD[Where+j] = d;
   }
   #ifdef OMP
     j = ParIdamin(NST, &pD[Where]);
   #else
     j = SeqIdamin(NST, &pD[Where]);
   #endif

   #ifdef TALK
     //printf("\t\tAt %d frame the postion of the minimum is %d\n", NSeq, j);
   #endif

   return j;
}



int DTWProcVar(const MyType *Sequence, const MyType *Costs, MyType * __restrict pD, const int NSeq, const int Where, const int NST)
{
   int NSTplusNC;
   int Ref;
   int Refpos[T_COSTS];
   int i, j;

   /* Some auxilar varibles */
   NSTplusNC = N_COSTS + NST;
   Ref = ((NSeq + N_COSTS) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   /* Referece index for each cost */
   for (i=0; i<N_COSTS; i++)
      Refpos[i] = Ref - i;

   for (i=N_COSTS; i<T_COSTS; i++)
      Refpos[i] = ((NSeq + (T_COSTS-i)) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(j=0; j<NST; j++)
   {
      MyType d, d2;
      int    k, Pos;

      #ifdef SIMPLE
         d = FLT_MAX;
      #else
         d = DBL_MAX;
      #endif

      for(k=0; k<T_COSTS; k++)
      {
         Pos = Refpos[k] + j;

         d2 = Sequence[j]*Costs[k] + pD[Pos];
         if (d2 < d) d=d2;
      }

      pD[Where+j] = d;
   }

   #ifdef OMP
     j = ParIdamin(NST, &pD[Where]);
   #else
     j = SeqIdamin(NST, &pD[Where]);
   #endif

   return j;
}


int DTWProcMed(const MyType *Sequence, const MyType *Costs, MyType * __restrict pD, const int NSeq, const int Where, const int NST,
                const int *states_corr, const int *times_state, SVelStates *SVT, const MyType WCOSTE)
{
   int NSTplusNC, NSTplusPad;
   int Ref;
   int Refpos[T_COSTS];
   int j, i;

   int salto[T_COSTS] = {1,2,3,4,1,1,1};

   /* Retrieve pointers from structure */
   int    *states_dur = SVT->states_dur;
   int    *pO         = SVT->onset_time;
   MyType *pV         = SVT->velocities;
   MyType *pM         = SVT->avg_vel;
   MyType *pS         = SVT->std_vel;
   int    *pL         = SVT->nvel;

   /* Some auxilar varibles */
   NSTplusNC = N_COSTS + NST;
   NSTplusPad = NSTplusNC * (N_COSTS + 1);
   Ref = ((NSeq + N_COSTS) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   /* Referece index for each cost */
   for (i=0; i<N_COSTS; i++)
      Refpos[i] = Ref - i;

   for (i=N_COSTS; i<T_COSTS; i++)
      Refpos[i] = ((NSeq + (T_COSTS-i)) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(j=0; j<NST; j++)
   {
      MyType d, d2;
      int    k, Pos, Posb;
      MyType vel;
      MyType prob;
      int    st, sta, stb;
      MyType coste;
      int idx;

      #ifdef SIMPLE
         d = FLT_MAX;
      #else
         d = DBL_MAX;
      #endif

      st = times_state[j];

      Posb = Refpos[0] + j;
      stb  = st;

      for(k=0; k<T_COSTS; k++)
      {
         Pos = Refpos[k] + j;
         coste = Costs[k];
         if (j-salto[k]<0)
            sta = -1;
         else
            sta = times_state[j-salto[k]];


        /*if (NSeq == NSEQTEST && j==JTEST)
             printf("numero vels = %d; el coste = %f\n", pL[Pos], coste);*/

         if (pL[Pos]>=N_MAXVEL)
         {
            vel   = CALCVEL(states_dur[sta],  NSeq-pO[Pos]+1+ANTCF);
            prob  = CDFGAUSS(vel, pM[Pos], pS[Pos]);
            coste = (st > sta) ? (Costs[k] + WCOSTE*prob) : (Costs[k] + WCOSTE*(1-prob));
            /*if (NSeq == NSEQTEST && j==JTEST)
                 printf("vel = %f; media = %f; var = %f; prob = %f; el coste = %f\n", vel, pM[Pos], pS[Pos], prob, coste);*/
         }

         d2 = Sequence[j]*coste + pD[Pos];
         if (d2 < d) {d=d2; stb=sta; Posb=Pos;}
      }

      idx = Where + j;
      pD[idx] = d;
      pO[idx] = pO[Posb];
      pL[idx] = pL[Posb];
      pM[idx] = pM[Posb];
      pS[idx] = pS[Posb];
      for (i=0; i<N_MAXVEL; i++)
         pV[idx + i*NSTplusPad] = pV[Posb + i*NSTplusPad];

      if (st>stb)
      {
         pO[idx] = NSeq;
         if (states_corr[st]) {
            pV[idx + (pL[idx]%N_MAXVEL) * NSTplusPad] = CALCVEL(states_dur[stb], NSeq-pO[Posb]);
            pL[idx] = 1 + pL[Posb];
            ComputeVelMed(pV, Posb, &pM[idx], &pS[idx], NSTplusPad); }
      }
   }

   #ifdef OMP
     j = ParIdamin(NST, &pD[Where]);
   #else
     j = SeqIdamin(NST, &pD[Where]);
   #endif

   return j;
}



void WriteAlignment(const int i, const int pos_min, int *stateprev, const int *preload, FILE *fp_aln)
{
   int    state, j;
   MyType t_est, fs = 44100;

   t_est = (MyType)((i-1)*TAMMUESTRA + TAMTRAMA/2)*1000/fs;

   state = preload[pos_min];
   if (state > *stateprev)
   {
      for (j = *stateprev+1; j <= state; j++)
         fprintf(fp_aln, "%d\t%f\t%f\n", j, t_est, t_est);
      *stateprev = state;
   }
}




int DTWProcMed2(const MyType *Sequence, const MyType *Costs, MyType * __restrict pD, const int NSeq, const int Where, const int NST,
               const int *states_corr, const int *times_state, SVelStates *SVT)
{
   int NSTplusNC, NSTplusPad;
   int Ref;
   int Refpos[T_COSTS];
   int j, i;

   /* Retrieve pointers from structure */
   int    *states_dur = SVT->states_dur;
   int    *pO         = SVT->onset_time;
   MyType *pV         = SVT->velocities;
   MyType *pA         = SVT->avg_vel;
   MyType *pS         = SVT->std_vel;
   int    *pL         = SVT->nvel;

   /* Some auxilar varibles */
   NSTplusNC = N_COSTS + NST;
   NSTplusPad = NSTplusNC * (N_COSTS + 1);
   Ref = ((NSeq + N_COSTS) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   /* Referece index for each cost */
   for (i=0; i<N_COSTS; i++)
      Refpos[i] = Ref - i;

   for (i=N_COSTS; i<T_COSTS; i++)
      Refpos[i] = ((NSeq + (T_COSTS-i)) % TBLOCK) * NSTplusNC + N_COSTS - 1;

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(j=0; j<NST; j++)
   {
      MyType d, d2;
      int    k, Pos, Posb;
      MyType vel;
      MyType prob;
      int    st, sta, stb;
      MyType avg, std;
      MyType coste;
      int idx;
      int i;

      #ifdef SIMPLE
         d = FLT_MAX;
      #else
         d = DBL_MAX;
      #endif

      Posb = Refpos[0] + j;
      stb  = 0;

      st  = times_state[j];

      for(k=0; k<N_COSTS; k++)
      {
         Pos = Refpos[k] + j;
         if (j-k-1<0)
            sta = -1;
         else
            sta = times_state[j-k-1];

         if (NSeq == NSEQTEST && j==JTEST)
             printf("Llevo %d valores\n", pL[Pos]);

         if (pL[Pos]>=N_MAXVEL)
         {
            ComputeVelMed(pV, Pos, &avg, &std, NSTplusPad);
            vel  = (MyType)states_dur[sta] / (MyType)(NSeq - pO[Pos] + 1);
            prob = 0.5 * erfc((avg-vel) * M_SQRT1_2 / std);

            if (NSeq == NSEQTEST && j==JTEST)
                printf("media = %f; std = %f; vel = %f; prob = %f; onset = %d (en %d); dur = %d\n", avg, std, vel, prob, pO[Pos], Pos, states_dur[sta]);

            if (st > sta)
               coste = 1 + 2*prob;
            else
               coste = 1 + 2*(1-prob);
         }
         else
         {
            coste = Costs[k];
         }

         d2 = Sequence[j]*coste + pD[Pos];
         if (d2 < d) {d=d2; stb=sta; Posb=Pos;}
      }

      for (k=N_COSTS; k<T_COSTS; k++)
      {
         Pos = Refpos[k] + j;
         if (j-1<0)
            sta = -1;
         else
            sta = times_state[j-1];

         if (pL[Pos]>=N_MAXVEL)
         {
            ComputeVelMed(pV, Pos, &avg, &std, NSTplusPad);
            vel  = states_dur[sta] / (NSeq - pO[Pos] + 1);
            prob = 0.5 * erfc((avg-vel) * M_SQRT1_2 / std);

            if (st > sta)
               coste = k-2 + 2*prob;
            else
               coste = k-2 + 2*(1-prob);
         }
         else
         {
            coste = Costs[k];
         }

         d2 = Sequence[j]*coste + pD[Pos];
         if (d2 < d) {d=d2; stb=sta; Posb=Pos;}
      }

      idx = Where + j;
      pD[idx] = d;
      pL[idx] = pL[Posb];
      for (i=0; i<N_MAXVEL; i++)
         pV[idx + i*NSTplusPad] = pV[Posb + i*NSTplusPad];

      if (st > stb)
      {
         pO[idx] = NSeq;
         if (states_corr[st]) {
            pV[idx + (pL[idx]%N_MAXVEL) * NSTplusPad] = states_dur[stb] / (NSeq - pO[Posb]);
            pL[idx] = 1 + pL[Posb]; }
      }
      else
      {
         pO[idx] = pO[Posb];
      }
      if (NSeq == NSEQTEST && j==JTEST)
          printf("onset actualizado %d guardado en %d\n", pO[idx], idx);

   }

   #ifdef OMP
     j = ParIdamin(NST, &pD[Where]);
   #else
     j = SeqIdamin(NST, &pD[Where]);
   #endif

   return j;
}



void ComputeVelMed(MyType *pV, int pos, MyType *med, MyType *var, int NSTplusPad)
{

   int i;
   MyType media = 0.0;
   MyType varianza = 0.0;

   for (i = 0; i < N_MAXVEL; i++) {
      media = pV[pos+i*NSTplusPad] + media;
   }
   media = media / N_MAXVEL;

   for (i = 0; i < N_MAXVEL; i++) {
      varianza = pow(pV[pos+i*NSTplusPad] - media, 2);
   }
   varianza = varianza / N_MAXVEL;

   *med = media;
   *var = sqrt(varianza);
}



/**
 *  \fn    void ApplyDist(const MyType *v_dxState, MyType *v_SxD, const int *I_SxD, const int size, const MyType ALPHA)
 *  \brief ApplyDist applies distortion
 *  \param v_dxState: (in) v_dxState vector
 *  \param v_SxD:    (out) v_SxD vector
 *  \param I_SxD:     (in) I_SxD vector
 *  \param size:      (in) Size of the vectors
 *  \param ALPHA:     (in) ALPHA value
 *  \return: Nothing, it is void
*/
void ApplyDist(const MyType *v_dxState, MyType *v_SxD, const int *I_SxD, const int size, const MyType ALPHA)
{
   int i;

   MyType vnorm=(MyType)0.0;

   #ifdef OMP
     #pragma omp parallel for reduction(+: vnorm)
   #endif
   for(i=0; i<size; i++)
   {
      v_SxD[i] = v_dxState[I_SxD[i]];
      vnorm += (v_SxD[i]*v_SxD[i]);
   }
   #ifdef SIMPLE
      vnorm = 1.0f / (sqrtf(vnorm) + FLT_EPSILON);

      /* ALPHA is defined as (-1.0)*ALPHA in function ReadParameters */
      #ifdef OMP
       #pragma omp parallel for
      #endif
      for(i=0;i<size;i++)
         v_SxD[i] = 1.0f - expf(ALPHA*fabsf(v_SxD[i]*vnorm));
   #else
      vnorm = 1.0 /  (sqrt(vnorm) + DBL_EPSILON);

      /* ALPHA is defined as (-1.0)*ALPHA in function ReadParameters */
      #ifdef OMP
        #pragma omp parallel for
      #endif
      for(i=0;i<size;i++)
         v_SxD[i] = 1.0 - exp(ALPHA*fabs(v_SxD[i]*vnorm));
   #endif
}


/**
 *  \fn    void ComputeDist(const MyType *v_cfreq, MyType * __restrict v_dxStates, MyType *tauxi, const MyType *norms, const MyType *s_fk, const MyType *ts_fk, const MyType BETA, const int nbases, const int nmidi)
 *  \brief ComputeDist computes distortion
 *  \param v_cfreq:     (in) v_cfreq vector
 *  \param v_dxStates: (out) v_dxStates vector
 *  \param tauxi:      (out) tauxi vector
 *  \param norms:       (in) norms vector
 *  \param s_fk:        (in) s_fk vector
 *  \param ts_fk:       (in) ts_fk vector
 *  \param BETA:        (in) BETA value
 *  \param nbases:      (in) Number of bases/combinations
 *  \param nmidi:       (in) Number of midi notes
 *  \return: Nothing, it is void
*/
void ComputeDist(const MyType *v_cfreq, MyType * __restrict v_dxStates, MyType *tauxi,
                 const MyType *norms, const MyType *s_fk, const MyType *ts_fk, const MyType BETA,
                 const int nbases, const int nmidi)
{
   int i;

   if (BETA>=(MyType)0.0 && BETA<=(MyType)0.0)
   {
      #ifdef OMP
        #pragma omp parallel for
      #endif
      for (i=0;i<nbases;i++)
      {
         int     j, itmp;
         MyType  A_kt, dsum, dtmp;

         itmp = i * N_MIDI_PAD;
         A_kt = (MyType)0.0;
         dsum = (MyType)0.0;

         /* BETA=0 --> a^(BETA-1) = a^-1 = 1/a */
         for (j=0; j<nmidi; j++)
            A_kt += v_cfreq[j] / s_fk[itmp+j];
         A_kt = A_kt / norms[i];

         for (j=0;j<nmidi;j++)
         {
            dtmp = v_cfreq[j] / (s_fk[itmp+j] * A_kt);
            #ifdef SIMPLE
               dsum += dtmp - logf(dtmp) - 1.0f;
            #else
               dsum += dtmp -  log(dtmp) - 1.0;
            #endif
         }
         v_dxStates[i] = dsum;
      }
   }
   else if (BETA>=(MyType)1.0 && BETA<=(MyType)1.0)
   {
      MyType A_kt=(MyType)0.0;

      /* BETA=1 --> a^(BETA-1) = a^0 = 1 due to a>=0. So next inner-loop */
      /* is moved here (out) because it not depend on index i, is always */
      /* the same result/operation/value                                 */
      #ifdef CBLAS
         #ifdef SIMPLE
            A_kt=cblas_sasum(nmidi, v_cfreq, 1);
         #else
            A_kt=cblas_dasum(nmidi, v_cfreq, 1);
         #endif
      #else
         for (i=0; i<nmidi; i++) { A_kt += v_cfreq[i]; }
      #endif

      #ifdef OMP
        #pragma omp parallel for
      #endif
      for (i=0;i<nbases;i++)
      {
         int     itmp, j;
         MyType  dsum, dtmp, dtmp2, dtmp3;

         itmp  = i * N_MIDI_PAD;
         dsum  = (MyType)0.0;
         dtmp2 = A_kt / norms[i];

         for (j=0;j<nmidi;j++)
         {
            dtmp = s_fk[itmp+j] * dtmp2;
            dtmp3= v_cfreq[j];
            #ifdef SIMPLE
               dsum += dtmp3*logf(dtmp3/dtmp) + dtmp - dtmp3;
            #else
               dsum += dtmp3* log(dtmp3/dtmp) + dtmp - dtmp3;
            #endif
         }
         v_dxStates[i] = dsum;
      }
   }
   else
   {
      MyType
        BetaMinusOne=BETA-(MyType)1.0,
        dConst=(MyType)1.0/(BETA*(BETA-(MyType)1.0));

      for (i=0; i<nmidi; i++)
         #ifdef SIMPLE
            tauxi[i]=powf(v_cfreq[i], BETA);
         #else
            tauxi[i]= pow(v_cfreq[i], BETA);
         #endif

      #ifdef OMP
        #pragma omp parallel for
      #endif
      for (i=0;i<nbases;i++)
      {
         int     j, itmp;
         MyType  A_kt, dsum, dtmp, dtmp2, dtmp3;

         itmp = i * N_MIDI_PAD;
         A_kt = (MyType)0.0;
         dsum = (MyType)0.0;

         #ifdef CBLAS
            #ifdef SIMPLE
               A_kt =cblas_sdot(nmidi, v_cfreq, 1,  &ts_fk[itmp], 1) / norms[i];
               dtmp3=powf(A_kt, BetaMinusOne);
            #else
               A_kt =cblas_ddot(nmidi, v_cfreq, 1,  &ts_fk[itmp], 1) / norms[i];
               dtmp3=pow(A_kt,BetaMinusOne);
            #endif
         #else
            for (j=0; j<nmidi; j++)
               A_kt += v_cfreq[j] * ts_fk[itmp+j];
            A_kt = A_kt / norms[i];
            #ifdef SIMPLE
               dtmp3=powf(A_kt, BetaMinusOne);
            #else
               dtmp3= pow(A_kt, BetaMinusOne);
            #endif
         #endif

         for (j=0;j<nmidi;j++)
         {
            dtmp  = s_fk[itmp+j]  * A_kt;
            dtmp2 = ts_fk[itmp+j] * dtmp3;
            dsum += (tauxi[j] + BetaMinusOne*dtmp2*dtmp - BETA*v_cfreq[j]*dtmp2)*dConst;
         }
         v_dxStates[i] = dsum;
      }
   }
}


/**
  *  \fn    ReadWavCPU1st(short *frame, FILE *fp)
  *  \brief ReadWavCPU1st reads first audio (frame) from WAV file when ARM is used
  *  \param frame: (out) Vector to store the first frame
  *  \param fp:     (in) ID of file with the information
  *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadWavCPU1st(short *frame, FILE *fp)
{
   if (fread(&frame[TAMMUESTRA], sizeof(short), TTminusTM, fp) != TTminusTM)
     return ErrReadFile;
   return OK;
}

/**
  *  \fn    ReadWavCPU(short *frame, FILE *fp)
  *  \brief ReadWavCPU reads current audio (frame) from WAV file when ARM is used
  *  \param frame: (out) Vector to store the current frame
  *  \param fp:     (in) ID of file with data
  *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadWavCPU(short *frame, FILE *fp)
{
   memmove(frame, &frame[TAMMUESTRA], sizeof(short)*TTminusTM);
   if (fread(&frame[TTminusTM], sizeof(short), TAMMUESTRA, fp) != TAMMUESTRA) return ErrReadFile;
   return OK;
}


#ifdef ALSA
  /**
    *  \fn    int ReadAlsaCPU1st(short *frame, snd_pcm_t *DeviceID, FILE *fpdump)
    *  \brief ReadAlsaCPU1st reads from microphone the first audio (frame) when CPU is used
    *  \param frame:    (out) Vector to store the audio of the first frame
    *  \param DeviceID:  (in) ALSA Sound devide identifier
    *  \param fpdump:   (out) File handle when DUMP is active
    *  \return: 0 if all is OK, otherwise a code error (see defines.h)
    *
  */
  int ReadAlsaCPU1st(short *frame, snd_pcm_t *DeviceID, FILE *fpdump)
  {
    if (snd_pcm_readi(DeviceID, &frame[TAMMUESTRA], TTminusTM) != TTminusTM) return ErrReadDevice;

    #ifdef DUMP
      if (fwrite(&frame[TAMMUESTRA], sizeof(short), TTminusTM, fpdump) != TTminusTM) return ErrWriteFile;
    #endif

    return OK;
  }

  /**
    *  \fn    int ReadAlsaCPU(short *frame, snd_pcm_t *DeviceID, FILE *fpdump)
    *  \brief ReadAlsaCPU reads from microphone the current audio (frame) when CPU is used
    *  \param frame:      (out) Vector to store the audio of the current frame
    *  \param DeviceID:    (in) ALSA Sound devide identifier
    *  \param fpdump:     (out) File handle when DUMP is active
    *  \return: 0 if all is OK, otherwise a code error (see defines.h)
    *
  */
  int ReadAlsaCPU(short *frame, snd_pcm_t *DeviceID, FILE *fpdump)
  {
    memmove(frame, &frame[TAMMUESTRA], sizeof(short)*TTminusTM);

    if (snd_pcm_readi(DeviceID, &frame[TTminusTM], TAMMUESTRA) != TAMMUESTRA) return ErrReadDevice;

    #ifdef DUMP
      if (fwrite(&frame[TTminusTM], sizeof(short), TAMMUESTRA, fpdump) != TAMMUESTRA) return ErrWriteFile;
    #endif

    return OK;
  }
#endif


/**
 *  \fn    void BetaNorm(MyType *v_cfreq, const int nmidi, MyType BETA)
 *  \brief BetaNorm normalizes v_cfreq vector to have beta-norm 1
 *  \param v_cfreq: (inout) v_cfreq vector
 *  \param BETA:       (in) BETA value
 *  \param nmidi:      (in) Number of midi notes
 *  \return: Nothing, it is void
*/
void BetaNorm(MyType *v_cfreq, const int nmidi, MyType BETA)
{
   /* nmidi is small and this procedure is only used at the beginning, we do not spend time with openmp and blas */
   int i;
   MyType value = 0.0;

   if (BETA>=(MyType)0.0 && BETA<=(MyType)0.0)
     /* function not defined. Changed to BETA=1.0 */
     BETA=1.0;

   if (BETA>=(MyType)1.0 && BETA<=(MyType)1.0)
   {
     #ifdef OMP
       #pragma omp parallel for reduction(+: value)
     #endif
     for(i=0; i<nmidi; i++)
       value += v_cfreq[i];
   }
   else
   {
     #ifdef SIMPLE
       #ifdef OMP
         #pragma omp parallel for reduction(+: value)
       #endif
       for(i=0; i<nmidi; i++)
         value += powf(v_cfreq[i], BETA);
       value = powf(value, 1.0/BETA);
     #else
       #ifdef OMP
         #pragma omp parallel for reduction(+: value)
       #endif
       for(i=0; i<nmidi; i++)
         value += pow(v_cfreq[i], BETA);
       value = pow(value, 1.0/BETA);
     #endif
   }

   #ifdef OMP
     #pragma omp parallel for
   #endif
   for(i=0; i<nmidi; i++)
     v_cfreq[i] = v_cfreq[i]/ value;
}


void BetaNormBien(MyType *v_cfreq, const int nmidi, MyType BETA)
{
   int i;
   MyType value = 0.0;

   if (BETA>=(MyType)0.0 && BETA<=(MyType)0.0)
     /* function not defined. Changed to BETA=1.0 */
     BETA=1.0;

   if (BETA>=(MyType)1.0 && BETA<=(MyType)1.0)
   {
     #ifdef OMP
       #pragma omp parallel for reduction(+: value)
     #endif
     for(i=0; i<nmidi; i++)
       value += v_cfreq[i];
   }
   else
   {
     #ifdef SIMPLE
       #ifdef OMP
         #pragma omp parallel for reduction(+: value)
       #endif
       for(i=0; i<nmidi; i++)
         value += powf(v_cfreq[i], BETA);
       value = powf(value, 1.0f/BETA);
     #else
       #ifdef OMP
         #pragma omp parallel for reduction(+: value)
       #endif
       for(i=0; i<nmidi; i++)
         value += pow(v_cfreq[i], BETA);
       value = pow(value, 1.0/BETA);
     #endif
   }

   if (value > (MyType)0.0)
   {
     #ifdef OMP
       #pragma omp parallel for
     #endif
     for(i=0; i<nmidi; i++)
       v_cfreq[i] = v_cfreq[i]/ value;
   }
   else
   {
     #ifdef SIMPLE
       value = powf(nmidi, -1.0f/BETA); // assumed BETA>0.0
     #else
       value = pow (nmidi, -1.0/BETA);  // assumed BETA>0.0
     #endif
     #ifdef OMP
       #pragma omp parallel for
     #endif
     for(i=0; i<nmidi; i++)
       v_cfreq[i] = value;
   }
}
