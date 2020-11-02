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
 *  \file    FileFunctions.c
 *  \brief   File with code of auxiliar functions using by ReMAS, both CPU and GPU.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "FileFunctions.h"

/**
 *  \fn    int ReadParameters(DTWconst *Param, DTWfiles *NameFiles, const char *filename)
 *  \brief ReadParameters reads ReMAS global parameters from file
 *  \param Param:     (out) Struct where read information (general ReMAS parameters) are stored
 *  \param NameFiles: (out) Struct where read filenames (with input/output general ReMAS info) are stored
 *  \param filename:   (in) Name of the file with general configuration input information for ReMAS
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadParameters(DTWconst *Param, DTWfiles *NameFiles, const char *filename)
{
   FILE *fp;

   int leidos=0, i=0;

   CHECKNULL(fp = fopen(filename,"r"));

   leidos += fscanf(fp, "%d\n", &Param->N_BASES);
   leidos += fscanf(fp, "%d\n", &Param->N_STATES);

   #ifdef SIMPLE
     leidos += fscanf(fp, "%f\n",  &Param->NoteLen);
     leidos += fscanf(fp, "%f\n",  &Param->ALPHA);
   #else
     leidos += fscanf(fp, "%lf\n", &Param->NoteLen);
     leidos += fscanf(fp, "%lf\n", &Param->ALPHA);
   #endif
   Param->ALPHA = -(Param->ALPHA);

   NameFiles->file_hanning     =(char *)malloc(1024);
   NameFiles->file_frame       =(char *)malloc(1024);
   NameFiles->file_partitura   =(char *)malloc(1024);
   NameFiles->file_kmax        =(char *)malloc(1024);
   NameFiles->file_kmin        =(char *)malloc(1024);
   NameFiles->fileStates_Time_e=(char *)malloc(1024);
   NameFiles->fileStates_Time_i=(char *)malloc(1024);
   NameFiles->fileStates_seq   =(char *)malloc(1024);
   NameFiles->fileStates_corr  =(char *)malloc(1024);

   leidos += fscanf(fp, "%s\n", NameFiles->file_hanning);
   leidos += fscanf(fp, "%s\n", NameFiles->file_frame);
   leidos += fscanf(fp, "%s\n", NameFiles->file_partitura);
   leidos += fscanf(fp, "%s\n", NameFiles->file_kmax);
   leidos += fscanf(fp, "%s\n", NameFiles->file_kmin);
   leidos += fscanf(fp, "%s\n", NameFiles->fileStates_Time_e);
   leidos += fscanf(fp, "%s\n", NameFiles->fileStates_Time_i);
   leidos += fscanf(fp, "%s\n", NameFiles->fileStates_seq);
   leidos += fscanf(fp, "%s\n", NameFiles->fileStates_corr);

   leidos += fscanf(fp, "%d\n", &Param->WAVorMIC);
   leidos += fscanf(fp, "%s\n",  Param->SoundID);
   leidos += fscanf(fp, "%d\n", &Param->Time_MIC);

   leidos += fscanf(fp, "%d\n", &Param->NCliOSC);
   if (Param->NCliOSC > MaxOSC) return ErrInfoReaded;

   for (i=0; i<Param->NCliOSC; i++)
   {   
      leidos += fscanf(fp, "%s\n", Param->HostIP[i]);
      leidos += fscanf(fp, "%s\n", Param->HostPort[i]);
   }
   fclose(fp);

   #ifdef DUMP
      printf("Number of Bases:      %d\n", Param->N_BASES);
      printf("Number of States:     %d\n", Param->N_STATES);
      printf("Note durarion (sec.): %f\n", Param->NoteLen);
      printf("ALPHA:                %f\n", Param->ALPHA);
      printf("Hanning       File:   %s\n", NameFiles->file_hanning);
      if (Param->WAVorMIC == 0)
         printf("Wav           File:   %s. USED\n", NameFiles->file_frame);
      else
         printf("Wav           File:   %s. NOT USED\n", NameFiles->file_frame);
      printf("Score         File:   %s\n", NameFiles->file_partitura);
      printf("KMAX          File:   %s\n", NameFiles->file_kmax);
      printf("KMIN          File:   %s\n", NameFiles->file_kmin);
      printf("States_time_e File:   %s\n", NameFiles->fileStates_Time_e);
      printf("States_time_i File:   %s\n", NameFiles->fileStates_Time_i);
      printf("States_seq    File:   %s\n", NameFiles->fileStates_seq);
      printf("States_correl File:   %s\n", NameFiles->fileStates_corr);
      if (Param->WAVorMIC != 0)
      {
         printf("Used sound device ID: %s\n", Param->SoundID);
         printf("Used sound MIDI time: %d\n", Param->Time_MIC);
      }
      printf("%d OSC clients are used. They are:\n", Param->NCliOSC);
      for (i=0; i<Param->NCliOSC; i++)
      {   
         printf("Host IP:   %s\n", Param->HostIP[i]);
         printf("Host Port: %s\n", Param->HostPort[i]);
      }
   #endif

   if (leidos != (17 + (Param->NCliOSC*2))) return ErrInfoReaded; else return OK;
}


/**
 *  \fn    int ReadVector(MyType *vector, const int size, const char *filename)
 *  \brief ReadVector fills a MyType vector with the MyType info stores in a file
 *  \param vector:  (out) Vector of type MyType (see defines.h) to store the data
 *  \param size:     (in) Number of elements to read
 *  \param filename: (in) Name of the file with the information
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadVector(MyType *vector, const int size, const char *filename)
{
   FILE *fp;

   CHECKNULL(fp = fopen(filename,"rb"));
   if (fread(vector, sizeof(MyType), size, fp) != size)  
     { fclose(fp); return ErrReadFile; }
   else
     { fclose(fp); return OK; }

/* MyType valor;
   int contLineas, leidos;

   CHECKNULL(fp = fopen(filename,"rb"));
     
   contLineas=0;

   leidos=fread(&valor, sizeof(MyType), 1, fp);
   if (leidos != 1) { fclose(fp); return ErrReadFile; }

   while (!feof(fp)) {
      if (contLineas<size) vector[contLineas]=valor;
      contLineas++;

      leidos=fread(&valor, sizeof(MyType), 1, fp);     
      if ((leidos != 1) && (!feof(fp))) { fclose(fp); return ErrReadFile; }
   }
   fclose(fp);
   if (contLineas != size) return ErrInfoReaded; else return OK;
*/
}


/**
 *  \fn    int ReadVectorInt64(int *vector, const int size, const char *filename)
 *  \brief ReadVectorInt64 fills a int vector with the int64 info stores in a file
 *  \param vector:  (out) Vector to store the data
 *  \param size:     (in) Number of elements to read
 *  \param filename: (in) Name of the file with the information
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadVectorInt64(int *vector, const int size, const char *filename)
{
   FILE *fp;
   int contLineas, leidos;
   
   // BY RANILLA 03-03-2016 13:54. Dirty, we need better solution
   #ifdef ARM32
     long long int valorLong;
     int nbytes=sizeof(long long int);
   #else
     long int valorLong;
     int nbytes=sizeof(long int);
   #endif

   CHECKNULL(fp=fopen(filename, "rb"));

   contLineas=0;

   leidos=fread(&valorLong, nbytes, 1, fp);
   if (leidos != 1) { fclose(fp); return ErrReadFile; } 
   
   while (!feof(fp))
   {
      if (contLineas < size)  vector[contLineas]=(int)valorLong;
      contLineas++;

      leidos=fread(&valorLong, nbytes, 1, fp);
      if ((leidos != 1) && (!feof(fp))) { fclose(fp); return ErrReadFile; }
   }
   fclose(fp);
   if (contLineas != size) return ErrInfoReaded; else return OK;

}


/**
 *  \fn    int ReadS_fk(MyType *s_fk, const int BASES, const char *filename)
 *  \brief ReadS_fk fills the vector s_fk with the info stores in a file
 *  \param s_fk:    (out) Vector s_fk to store the data
 *  \param BASES:    (in) Number of BASES
 *  \param filename: (in) Name of the file with the information
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int ReadS_fk(MyType *s_fk, const int BASES, const char *filename)
{
   long i, k;

   long size = N_MIDI_PAD*BASES;
   MyType data;

   FILE *fp;

   CHECKNULL(fp=fopen(filename, "rb"));

   i=0;
   k=fread(&data, sizeof(MyType), 1, fp);
   if (k != 1) { fclose(fp); return ErrReadFile; }

   while(!feof(fp))
   {
      if (i<size)
      {
        s_fk[i]=data;
        if ((i%N_MIDI_PAD)< (N_MIDI-1))
           i++;
        else
           i += (N_MIDI_PAD-N_MIDI+1);
      }

      k=fread(&data, sizeof(MyType), 1, fp);
      if ((k != 1) && (!feof(fp))) { fclose(fp); return ErrReadFile; }
   }
   fclose(fp);
   if (i != size) return ErrInfoReaded; else return OK;
}


/**
 *  \fn    void FreeFiles(DTWfiles *NameFiles)
 *  \brief FreeFiles frees the reserved memory of a struct
 *  \param NameFiles: (inout) Name os the struct
 *  \return: Nothing, it is void
*/
void FreeFiles(DTWfiles *NameFiles)
{
   free(NameFiles->file_hanning);
   free(NameFiles->file_frame);
   free(NameFiles->file_partitura);
   free(NameFiles->file_kmax);
   free(NameFiles->file_kmin);
   free(NameFiles->fileStates_Time_e);
   free(NameFiles->fileStates_Time_i);
   free(NameFiles->fileStates_seq);
   free(NameFiles->fileStates_corr);
}

