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
 *  \file    TempoFunctions.h
 *  \brief   File with TEMPO and auxiliar functions using by ReMAS, both CPU and GPU.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/

#pragma once

#ifndef ATEMPO_H
#define ATEMPO_H

#include "defines.h"

void pqsort      (MyType *, int);
void pqsort_inner(MyType *, int, int, int);

int    cmpfunc(const void *, const void *);
MyType TheilSenRegression(int *, int *, int, bool *);

#ifdef OSC
  #include <NetFunctions.h>
  int ComputeTempoOSC  (STempo *,   int, int, int, int *, lo_address *, int, int);
  int ComputeTempoOSCRL(STempoRL *, int, int, int, int *, lo_address *, int);
#endif
void ComputeTempo  (STempo *,   int, int, int, int *, int);
void ComputeTempoRL(STempoRL *, int, int, int, int *);

//Size, weights and speeds (speed=1.0 at start time) for gaussian filter
//#define SizeGauss 3
//static const MyType WeighGauss[SizeGauss]={0.10, 0.36, 0.54};
//static       MyType SpeedGauss[SizeGauss]={1.00, 1.00, 1.00};
#define SizeGauss 5
static const MyType WeighGauss[SizeGauss]={0.10, 0.15, 0.20, 0.25, 0.30};
static       MyType SpeedGauss[SizeGauss]={1.00, 1.00, 1.00, 1.00, 1.00};


#ifdef OMP
  #include <omp.h>
  /**
   *  \fn    void  pqsort(MyType *v, int size)
   *  \brief pqsort performs parallel quicksort using openmp 
   *  \param v:    (inout) the vector
   *  \param size:    (in) the vector size
   *  \return: None, it is void
  */
  void pqsort(MyType *v, int size)
  {
    #pragma omp parallel
    {
      #pragma omp single nowait
        { pqsort_inner(v, 0, size-1, size/omp_get_num_threads()); }
    }
  }


  /**
   *  \fn    void  pqsort_inner(MyType *v, int left, int right, int cutoff)
   *  \brief pqsort_inner performs the internal parallel quicksort using openmp 
   *  \param v:     (inout) the vector
   *  \param left:     (in) the left vector side
   *  \param right:    (in) the right vector side
   *  \param cutoff:   (in) cutoff point
   *  \return: None, it is void
  */
  void pqsort_inner(MyType *v, int left, int right, int cutoff)
  {
    int i=left, j=right;
    MyType tmp, pivot=v[(left + right) / 2];
	
    while (i <= j)
    {
      while (v[i] < pivot) i++;
      while (v[j] > pivot) j--;

      if (i <= j) { tmp = v[i]; v[i] = v[j]; v[j] = tmp; i++; j--; }
    }

    if (((right-left)<cutoff)) {
      if (left < j)  { pqsort_inner(v, left, j,  cutoff); }
      if (i < right) { pqsort_inner(v, i, right, cutoff); }
    } else {
      #ifdef OMP4
        #pragma omp task untied mergeable
          { pqsort_inner(v, left, j, cutoff); }
        #pragma omp task untied mergeable
          { pqsort_inner(v, i, right, cutoff); }
      #else
        #pragma omp task untied
          { pqsort_inner(v, left, j, cutoff); }
        #pragma omp task untied
          { pqsort_inner(v, i, right, cutoff); }      
      #endif
    }
  }
#else
  /**
   *  \fn    int cmpfunc(const void * a, const void * b)
   *  \brief cmpfunc compares two values.
   *  \param a: (in) First value
   *  \param b: (in) Second value
   *  \return: comparison result
  */
  int cmpfunc(const void *a, const void *b)
  {
     if (*(MyType *)a < *(MyType *)b)
       { return -1; }
     else if (*(MyType *)a == *(MyType *)b)
       { return 0; }
     else
       { return 1; }
  }
#endif


/**
 *  \fn    MyType TheilSenRegression(int *x, int *y, int L, bool *outlier)
 *  \brief TheilSenRegression computes Theil-Sen linear regression.
 *  \param x:        (in) Audio time of anchor points
 *  \param y:        (in) Score time of anchor points
 *  \param L:        (in) Number of the last anchor point
 *  \param outlier: (out) 1 if last anchor point is an outlier, 0 otherwise
 *  \return: Estimated slope
*/
MyType TheilSenRegression(int *x, int *y, int L, bool *outlier)
{
  MyType  a=.0, b=.0, lerr=.0, slopes[NUMAPPAIRS], err[NUMAPMAX];
  int     i, j, N, M, idx=0;

  if (L > NUMAPMAX) { N=NUMAPMAX; } else { N=L; }

  M=(N*(N-1))/2;

  /* slope between pairs */
  #ifdef OMP
    #pragma omp parallel for private(j, idx)
      for (i=0; i<N; i++)
      {
        idx=i*N-((i+2)*(i+1)/2);
        for (j=i+1; j<N; j++)
          slopes[idx+j] = (MyType)(y[j]-y[i]) / (MyType)(x[j]-x[i]);
      }
  #else
    for (i = 0; i < N; i++)
      for (j = i+1; j < N; j++)
      {
        slopes[idx] = (MyType)(y[j]-y[i]) / (MyType)(x[j]-x[i]);
        idx++;
      }
  #endif

  /* median (slow sorting method) */
  #ifdef OMP
    pqsort(slopes, M);
  #else
    qsort (slopes, M, sizeof(MyType), cmpfunc);
  #endif
  
  if (M%2 == 0)
    { a = (slopes[M/2-1] + slopes[M/2]) / 2; }
  else
    { a = slopes[(M-1)/2]; }
  if ((a < 0.8) || (a > 1.2))  { *outlier = 1; return a; }

  #ifdef OMP
    #pragma omp parallel for reduction(+: b)
  #endif
  for (i=0; i<N; i++)
    b = b + (y[i] - a*x[i]);
  b = b/N;

  /* fitting error for each point */
  #ifdef OMP4
    #pragma omp simd
  #else
    #ifdef OMP
      #pragma omp parallel for reduction(+: b)
    #endif
  #endif
  for (i = 0; i < N; i++)
    #ifdef SIMPLE
      err[i] = fabsf(y[i] - (a*x[i] + b));
    #else
      err[i] = fabs (y[i] - (a*x[i] + b));
    #endif

  /* check if last point is an outlier */
  lerr = err[(L-1)%NUMAPMAX];
  #ifdef OMP
    pqsort(err, N);
  #else
    qsort (err, N, sizeof(MyType), cmpfunc);
  #endif
  if (N%2 == 0)
    { *outlier = (lerr > err[N/2-1]); }
  else
    { *outlier = (lerr > err[(N-1)/2]); }

  return a;
}


#ifdef OSC
  /**
   *  \fn    int ComputeTempoOSC(STempo *TEMPO, int current, int pos_min, int count_min, int *states_corr, lo_address *DirOSC, int NCli, int NStates)
   *  \brief ComputeTempoOSC calculates the tempo for current frame using gaussian filter and OSC
   *  \param TEMPO:      (inout) Struct for control the tempo
   *  \param current:       (in) Current frame
   *  \param pos_min:       (in) The position of the minimum in Current frame
   *  \param count_min:     (in) Times pos_min verify one hypothesis over vector states_time_e
   *  \param states_corr:   (in) states_corr vector
   *  \param DirOSC         (in) OSC clientes IP direction
   *  \param NCli           (in) Number of OSC clients
   *  \param NStates:       (in) Number of states
   * \return: OK if all Ok. Otherwise an error code
   *
  */
  int ComputeTempoOSC(STempo *TEMPO, int current, int pos_min, int count_min, int *states_corr, lo_address *DirOSC, int NCli, int NStates)
  {
    int i;
    MyType nextScoreTime, Speed;
   
    if ((count_min > TEMPO->PrevState) && (states_corr[count_min]==1) && (count_min < (NStates-1)))
    {
      /* Vmatch */
      Speed=((pos_min - TEMPO->MidiFrame) * TrainJumpTime) / ((current - TEMPO->RealFrame) * RunJumpTime);

      TEMPO->SynthTime = ((current - TEMPO->RealFrame) * TrainJumpTime) * TEMPO->SynthSpeed + TEMPO->SynthTime;

      nextScoreTime = DelayTimeMidi * Speed + pos_min * TrainJumpTime;
   
      TEMPO->SynthSpeed = (nextScoreTime - TEMPO->SynthTime) / DelayTimeMidi;

      //TEMPO->SoloSpeed = Speed;
      Speed=0.0;
      for (i=0; i<SizeGauss-1; i++)
      {
        Speed += SpeedGauss[i]*WeighGauss[i];
        SpeedGauss[i]= SpeedGauss[i+1];
      }
      Speed += SpeedGauss[SizeGauss-1]*WeighGauss[SizeGauss-1];

      if (TEMPO->SynthSpeed < 0)
        { TEMPO->SynthSpeed = 0.25; }
      else if (TEMPO->SynthSpeed > (1.2*Speed))
        { TEMPO->SynthSpeed = (1.2*Speed); }
      else if (TEMPO->SynthSpeed < (0.8*Speed))
        { TEMPO->SynthSpeed = (0.8*Speed); }   
      SpeedGauss[SizeGauss-1]=TEMPO->SynthSpeed;

      TEMPO->MidiFrame=pos_min;
      TEMPO->RealFrame=current;
      TEMPO->PrevState=count_min;

      TEMPO->SoloSpeed = Speed;
      TEMPO->NextFrame = current + round((TEMPO->SynthTime - pos_min*TrainJumpTime)/(TEMPO->SoloSpeed - TEMPO->SynthSpeed) / TrainJumpTime);
      #ifdef TALK
        printf("Vmatch: frame %d pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
      #endif
      for (i=0; i<NCli; i++) CHECKERR(SendTempo(DirOSC[i], TEMPO->SynthSpeed*100));
    }
    else if (TEMPO->NextFrame == current) {
      /* Vtempo */
      TEMPO->SynthSpeed = TEMPO->SoloSpeed;
      #ifdef TALK
        printf("Vtempo: matching with the soloist. Frame %d, pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
      #endif
      for (i=0; i<NCli; i++) CHECKERR(SendTempo(DirOSC[i], TEMPO->SynthSpeed*100));
    }
    return OK;
  }


  /**
   *  \fn    int ComputeTempoOSCRL(STempoRL *TEMPO, int current, int pos_min, int count_min, int *states_corr, lo_address *, int NCli)
   *  \brief ComputeTempoOSCRL calculates tempo and controls synthesizer speed using linear regression and OSC
   *  \param TEMPO:    (inout) Struct for control the tempo
   *  \param current:     (in) Current frame
   *  \param pos_min:     (in) Estimated score position (frame) in Current frame
   *  \param count_min:   (in) Estimated score position (state) in Current frame
   *  \param states_corr: (in) states_corr vector
   *  \param DirOSC       (in) OSC client IP direction
   *  \param NCli         (in) Number of OSC clients
   *  \return: OK if all Ok. Otherwise an error code
   *
  */
  int ComputeTempoOSCRL(STempoRL *TEMPO, int current, int pos_min, int count_min, int *states_corr, lo_address *DirOSC, int NCli)
  {
    int    i;
    MyType SoloTime, TimeDiff, SpeedDiff, a;
    bool   outlier;

    /* Update synthesizer position */
    TEMPO->SynthTime = TEMPO->SynthTime + TEMPO->SynthSpeed*(MyType)RunJumpTime;

    /* Check if anchor point */
    if ((count_min > TEMPO->PrevState) && (states_corr[count_min]==1))
    {
       /* Store anchor point */
       TEMPO->AudioTimeAP[TEMPO->numap % NUMAPMAX] = current;
       TEMPO->ScoreTimeAP[TEMPO->numap % NUMAPMAX] = pos_min;
       TEMPO->numap++;
       TEMPO->PrevState = count_min;

       /* Performer position */
       SoloTime = (MyType)pos_min * (MyType)TrainJumpTime;
       #ifdef SIMPLE
         TimeDiff = fabsf(TEMPO->SynthTime - SoloTime);
       #else
         TimeDiff = fabs (TEMPO->SynthTime - SoloTime);
       #endif

       /* Performer speed */
       if ((TEMPO->matched) && (TimeDiff>0.050))
       {
          a = TheilSenRegression(TEMPO->AudioTimeAP, TEMPO->ScoreTimeAP, TEMPO->numap, &outlier);
          if (!outlier)
          {
             /* Vtempo */
             TEMPO->SoloSpeed = a;

             /* Vmatch: synth speed to match performer in DelayTimeMidi score secs */
             TEMPO->SynthSpeed = TEMPO->SoloSpeed * (SoloTime + DelayTimeMidi - TEMPO->SynthTime) / DelayTimeMidi;
             if (TEMPO->SynthSpeed < (MyType)0.1)
               TEMPO->SynthSpeed = (MyType)0.1;
             else if (TEMPO->SynthSpeed > (MyType)3.0)
               TEMPO->SynthSpeed = (MyType)3.0;

             /* Send Vmatch to the synthesizer (conviene bajar volumen) */
             for (i=0; i<NCli; i++) CHECKERR(SendTempo(DirOSC[i], TEMPO->SynthSpeed*100));

             /* Schedule next speed command */
             #ifdef SIMPLE
               SpeedDiff = fabsf(TEMPO->SoloSpeed - TEMPO->SynthSpeed);
               TEMPO->NextFrame = current + (int)roundf(TimeDiff / SpeedDiff / (MyType)RunJumpTime);
             #else
               SpeedDiff = fabs(TEMPO->SoloSpeed - TEMPO->SynthSpeed);
               TEMPO->NextFrame = current + (int)round (TimeDiff / SpeedDiff / (MyType)RunJumpTime);
             #endif
             TEMPO->matched = 0;
             #ifdef TALK
               printf("Vmatch: frame %d pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
             #endif
          }
       }
    }
    /* Check if NextFrame */
    if (TEMPO->NextFrame == current)
    {
       /* Send Vtempo to the synthesizer */
       TEMPO->SynthSpeed = TEMPO->SoloSpeed;
       TEMPO->matched = 1;
       for (i=0; i<NCli; i++) CHECKERR(SendTempo(DirOSC[i], TEMPO->SynthSpeed*100));
       #ifdef TALK
         printf("Vtempo: matching with the soloist. Frame %d, pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
       #endif
    }
    return OK;
  }
#endif


/**
 *  \fn    void ComputeTempo(STempo *TEMPO, int current, int pos_min, int count_min, int *states_corr, int NStates)
 *  \brief ComputeTempo calculates the tempo for current frame using gaussian filter
 *  \param TEMPO:      (inout) Struct for control the tempo
 *  \param current:       (in) Current frame
 *  \param pos_min:       (in) The position of the minimum in Current frame
 *  \param count_min:     (in) Times pos_min verify one hypothesis over vector states_time_e
 *  \param states_corr:   (in) states_corr vector
 *  \param NStates:       (in) Number of states
 * \return: None, it is void
 *
*/
void ComputeTempo(STempo *TEMPO, int current, int pos_min, int count_min, int *states_corr, int NStates)
{
  int i;
  MyType nextScoreTime, Speed;
  
  if ((count_min > TEMPO->PrevState) && (states_corr[count_min]==1) && (count_min < (NStates-1)))   
  {
      /* Vmatch */
      Speed=((pos_min - TEMPO->MidiFrame) * TrainJumpTime) / ((current - TEMPO->RealFrame) * RunJumpTime);

      TEMPO->SynthTime = ((current - TEMPO->RealFrame) * TrainJumpTime) * TEMPO->SynthSpeed + TEMPO->SynthTime;

      nextScoreTime = DelayTimeMidi * Speed + pos_min * TrainJumpTime;
   
      TEMPO->SynthSpeed = (nextScoreTime - TEMPO->SynthTime) / DelayTimeMidi;

      //TEMPO->SoloSpeed = Speed;
      Speed=0.0;
      for (i=0; i<SizeGauss-1; i++)
      {
        Speed += SpeedGauss[i]*WeighGauss[i];
        SpeedGauss[i]= SpeedGauss[i+1];
      }
      Speed += SpeedGauss[SizeGauss-1]*WeighGauss[SizeGauss-1];

      if (TEMPO->SynthSpeed < 0)
        { TEMPO->SynthSpeed = 0.25; }
      else if (TEMPO->SynthSpeed > (1.2*Speed))
        { TEMPO->SynthSpeed = (1.2*Speed); }
      else if (TEMPO->SynthSpeed < (0.8*Speed))
        { TEMPO->SynthSpeed = (0.8*Speed); }   
      SpeedGauss[SizeGauss-1]=TEMPO->SynthSpeed;

      TEMPO->MidiFrame=pos_min;
      TEMPO->RealFrame=current;
      TEMPO->PrevState=count_min;

      TEMPO->SoloSpeed = Speed;
      TEMPO->NextFrame = current + round((TEMPO->SynthTime - pos_min*TrainJumpTime)/(TEMPO->SoloSpeed - TEMPO->SynthSpeed) / TrainJumpTime);
      #ifdef TALK
        printf("Vmatch: frame %d pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
      #endif
  }
  else if (TEMPO->NextFrame == current)
  {
      /* Vtempo */
      TEMPO->SynthSpeed = TEMPO->SoloSpeed;
      #ifdef TALK
        printf("Vtempo: matching with the soloist. Frame %d, pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
      #endif
  }
}


/**
 *  \fn    void ComputeTempoRL(STempoRL *TEMPO, int current, int pos_min, int count_min, int *states_corr)
 *  \brief ComputeTempoRL calculates tempo and controls synthesizer speed using linear regression
 *  \param TEMPO:    (inout) Struct for control the tempo
 *  \param current:     (in) Current frame
 *  \param pos_min:     (in) Estimated score position (frame) in Current frame
 *  \param count_min:   (in) Estimated score position (state) in Current frame
 *  \param states_corr: (in) states_corr vector
 *  \return: None, it is void
 *
*/
void ComputeTempoRL(STempoRL *TEMPO, int current, int pos_min, int count_min, int *states_corr)
{
   MyType SoloTime, TimeDiff, SpeedDiff, a;
   bool   outlier;

   /* Update synthesizer position */
   TEMPO->SynthTime = TEMPO->SynthTime + TEMPO->SynthSpeed*(MyType)RunJumpTime;

   /* Check if anchor point */
   if ((count_min > TEMPO->PrevState) && (states_corr[count_min]==1))
   {
      /* Store anchor point */
      TEMPO->AudioTimeAP[TEMPO->numap % NUMAPMAX] = current;
      TEMPO->ScoreTimeAP[TEMPO->numap % NUMAPMAX] = pos_min;
      TEMPO->numap++;
      TEMPO->PrevState = count_min;

      /* Performer position */
      SoloTime = (MyType)pos_min * (MyType)TrainJumpTime;
      #ifdef SIMPLE
        TimeDiff = fabsf(TEMPO->SynthTime - SoloTime);
      #else
        TimeDiff = fabs (TEMPO->SynthTime - SoloTime);
      #endif

      /* Performer speed */
      if ((TEMPO->matched) && (TimeDiff>0.050))
      {
         a = TheilSenRegression(TEMPO->AudioTimeAP, TEMPO->ScoreTimeAP, TEMPO->numap, &outlier);
         if (!outlier)
         {
            /* Vtempo */
            TEMPO->SoloSpeed = a;

            /* Vmatch: synth speed to match performer in DelayTimeMidi score secs */
            TEMPO->SynthSpeed = TEMPO->SoloSpeed * (SoloTime + DelayTimeMidi - TEMPO->SynthTime) / DelayTimeMidi;
            if (TEMPO->SynthSpeed < (MyType)0.1)
              TEMPO->SynthSpeed = (MyType)0.1;
            else if (TEMPO->SynthSpeed > (MyType)3.0)
              TEMPO->SynthSpeed = (MyType)3.0;

            /* Schedule next speed command */
            #ifdef SIMPLE
              SpeedDiff = fabsf(TEMPO->SoloSpeed - TEMPO->SynthSpeed);
              TEMPO->NextFrame = current + (int)roundf(TimeDiff / SpeedDiff / (MyType)RunJumpTime);
            #else
              SpeedDiff = fabs(TEMPO->SoloSpeed - TEMPO->SynthSpeed);
              TEMPO->NextFrame = current + (int)round (TimeDiff / SpeedDiff / (MyType)RunJumpTime);
            #endif
            TEMPO->matched = 0;
            #ifdef TALK
              printf("Vmatch: frame %d pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
            #endif
         }
      }
   }
   /* Check if NextFrame */
   if (TEMPO->NextFrame == current)
   {
      /* Send Vtempo to the synthesizer */
      TEMPO->SynthSpeed = TEMPO->SoloSpeed;
      TEMPO->matched = 1;
      #ifdef TALK
        printf("Vtempo: matching with the soloist. Frame %d, pos_min %d SynthSpeed %1.5f\n", current, pos_min, TEMPO->SynthSpeed);
      #endif
   }
}

#endif