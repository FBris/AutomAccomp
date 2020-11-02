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
 *  \file    SoundFunctions.h
 *  \brief   File with sound (using ALSA) and auxiliar functions using by ReMAS, both CPU and GPU.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/


#pragma once

#ifndef ASOUND_H
#define ASOUND_H

#include "defines.h"
#include <strings.h>

#ifdef ALSA
  #include <asoundlib.h>
  snd_pcm_uframes_t SetMicParams(snd_pcm_t **, DTWconst);

  /**
   *  \fn    snd_pcm_uframes_t SetMicParams(snd_pcm_t **DeviceID, DTWconst Param)
   *  \brief SetMicParams configure the input sound device (microphone)
   *  \param DeviceID: (out) ALSA Sound devide identifier
   *  \param Param:     (in) Struct where general ReMAS parameters are stored
   *  \return: The input sound device buffer size
   *
  */
  snd_pcm_uframes_t SetMicParams(snd_pcm_t **DeviceID, DTWconst Param)
  {
     snd_pcm_hw_params_t *HwParams;
     snd_pcm_uframes_t    BufferSize=0;
     snd_pcm_format_t     format;
     snd_pcm_access_t     acces;
     unsigned int         tmp, rrate;

     /* Blocking aperture, last parameter=0 */
     CHECKERR(snd_pcm_open(DeviceID, Param.SoundID, SND_PCM_STREAM_CAPTURE, 0));

     /* Memory for hardware parameters structure */
     CHECKERR(snd_pcm_hw_params_malloc(&HwParams));

     /* Fill hardware structure with all permitted values */
     CHECKERR(snd_pcm_hw_params_any((*DeviceID), HwParams));

     /* Set access mode */
     CHECKERR(snd_pcm_hw_params_set_access((*DeviceID), HwParams, AlsaAccessMode));

     /* Set format */
     CHECKERR(snd_pcm_hw_params_set_format((*DeviceID), HwParams, AlsaAccessFormat));

     /* Set number input channels */
     CHECKERR(snd_pcm_hw_params_set_channels((*DeviceID), HwParams, AlsaChannels));

     /* Set rate */
     rrate=AlsaRate;
     CHECKERR(snd_pcm_hw_params_set_rate_near((*DeviceID), HwParams, &rrate, 0));

     /* Apply the hardware parameters that we've set. */
     CHECKERR(snd_pcm_hw_params((*DeviceID), HwParams));

     /* Get the buffer size */
     CHECKERR(snd_pcm_hw_params_get_buffer_size(HwParams, &BufferSize));

     /* Check all is OK */
     if (rrate != AlsaRate) CHECKERR(ErrAlsaHw);
     #ifdef TALK
       printf("Rate %u\n", rrate);
     #endif

     CHECKERR(snd_pcm_hw_params_get_access(HwParams, &acces));
     if (acces != AlsaAccessMode) CHECKERR(ErrAlsaHw);
     #ifdef TALK
       printf("Access %u\n", (unsigned int)acces);
     #endif

     CHECKERR(snd_pcm_hw_params_get_format(HwParams, &format));
     if (format != AlsaAccessFormat) CHECKERR(ErrAlsaHw);
     #ifdef TALK
       printf("Format %u\n", (unsigned int)format);
     #endif

     CHECKERR(snd_pcm_hw_params_get_channels(HwParams, &tmp));
     if (tmp != AlsaChannels) CHECKERR(ErrAlsaHw);
     #ifdef TALK
       printf("Channels number %u\n", tmp);
       printf("Buffer Size %lu\n", BufferSize);
     #endif

     /* Free the hardware parameters now */
     snd_pcm_hw_params_free(HwParams);

     return BufferSize;
  }
#endif

int Read_WAVHeader (WAVHeader *, FILE *);
/**
 *  \fn    int Read_WAVHeader(WAVHeader *Header, FILE *fp)
 *  \brief Read_WAVHeader reads header of a WAVE file, checks its compability and fill Header struct.
 *  \param Header: (out) Struct to fill
 *  \param fp:      (in) file ID with the information
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
*/
int Read_WAVHeader(WAVHeader *Header, FILE *fp)
{
   unsigned char buffer[4];

   /* Read string RIFF */
   if (fread(Header->riff, sizeof(Header->riff), 1, fp) != 1) return ErrReadFile;
   #ifdef DUMP
      printf("(01-04) riff string: %s\n", Header->riff);
   #endif

   /* 1st Read overall size, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 4, 1, fp) != 1) return ErrReadFile;
   Header->size = buffer[0] | (buffer[1]<<8) | (buffer[2]<<16) | (buffer[3]<<24);
   #ifdef DUMP
      printf("(05-08) Overall file size %u bytes\n", Header->size);
   #endif

   /* Read string WAVE */
   if (fread(Header->wave, sizeof(Header->wave), 1, fp) != 1) return ErrReadFile;
   #ifdef DUMP
      printf("(09-12) wave string: %s\n", Header->wave);
   #endif

   /* Read string FMT */
   if (fread(Header->fmt, sizeof(Header->fmt), 1, fp) != 1) return ErrReadFile;
   #ifdef DUMP
      printf("(13-16) fmt string: %s\n", Header->fmt);
   #endif

   /* 1st Read length of the format data, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 4, 1, fp) != 1) return ErrReadFile;
   Header->fmt_length = buffer[0] | (buffer[1]<<8) | (buffer[2]<<16) | (buffer[3]<<24);
   #ifdef DUMP
      printf("(17-20) Length of the format data: %u bytes\n", Header->fmt_length);
   #endif
   if (Header->fmt_length != 16)
   {
      printf("Fmt data length not equal to 16, not supported. End\n");
      return ErrWavFormat;
   }

   /* 1st Read format type, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 2, 1, fp) != 1) return ErrReadFile;
   Header->format_type = buffer[0] | (buffer[1] << 8);
   #ifdef DUMP
      printf("(21-22) Format type: %u (1 is PCM) \n", Header->format_type);
   #endif
   if (Header->format_type != 1)
   {
      printf("Format type not equal to PCM, not supported. End\n");
      return ErrWavFormat;
   }

   /* 1st Read number of channels, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 2, 1, fp) != 1) return ErrReadFile;
   Header->channels = buffer[0] | (buffer[1] << 8);
   #ifdef DUMP
      printf("(23-24) Number of channels: %u\n", Header->channels);
   #endif
   if (Header->channels != 1)
   {
      printf("Number of channels not equal to 1 (mono), not supported. End\n");
      return ErrWavFormat;
   }

   /* 1st Read sample_rate, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 4, 1, fp) != 1) return ErrReadFile;
   Header->sample_rate = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24);
   #ifdef DUMP
      printf("(25-28) Sample rate: %u\n", Header->sample_rate);
   #endif

   /* 1st Read byte rate, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 4, 1, fp) != 1) return ErrReadFile;
   Header->byte_rate = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24);
   #ifdef DUMP
      printf("(29-32) Byte Rate: %u\n", Header->byte_rate);
   #endif

   /* 1st Read block align, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 2, 1, fp) != 1) return ErrReadFile;
   Header->block_align = buffer[0] | (buffer[1] << 8);
   #ifdef DUMP
      printf("(33-34) Block Alignment: %u\n", Header->block_align);
   #endif

   /* 1st Read bits per sample, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 2, 1, fp) != 1) return ErrReadFile;
   Header->bits_per_sample = buffer[0] | (buffer[1] << 8);
   #ifdef DUMP
      printf("(35-36) Bits per sample: %u\n", Header->bits_per_sample);
   #endif
   if (Header->byte_rate != (Header->sample_rate * Header->channels * Header->bits_per_sample / 8))
   {
      printf("byte rate is not equal to (sample_rate * channels * bits_per_sample), not supported. End\n");
      return ErrWavFormat;
   }

   /* Read string DATA */
   /* Standard wav file header size is 44 bytes. Some audios have undocumented data   */
   /* into the header. The way to know if this is a not a 44 byte header is to check  */
   /* if strinf DATA is here                                                          */
   if (fread(Header->data_header, sizeof(Header->data_header), 1, fp) != 1) return ErrReadFile;
   #ifdef DUMP
      printf("(37-40) String DATA: %s\n", Header->data_header);
   #endif
   if (strncasecmp("data", Header->data_header, 4) != 0)
   {
      printf("Not standard wav file header size (44 bytes), not supported. End\n");
      return ErrWavFormat;
   }

   /* 1st Read data size, 2nd convert little endian -> big endian (4 byte int) */
   if (fread(buffer, 4, 1, fp) != 1) return ErrReadFile;
   Header->data_size = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24);
   #ifdef DUMP
      printf("(41-44) size of data section is %u bytes\n", Header->data_size);
   #endif

   Header->num_samples = Header->data_size / Header->block_align;
   #ifdef DUMP
      printf("Number of samples: %lu\n", Header->num_samples);
   #endif
   if (Header->num_samples != (8*Header->data_size) / (Header->channels*Header->bits_per_sample))
   {
      printf("(data_size / block_align) not equal to ((8*data_size) / (channels*bits_per_sample)). End\n");
      return ErrWavFormat;
   }

   Header->bytes_per_sample = (Header->bits_per_sample * Header->channels) / 8;
   #ifdef DUMP
      printf("Number of bytes per sample: %u\n", Header->bytes_per_sample);
   #endif
   if (Header->bytes_per_sample != 2)
   {
      printf("Byte per sample is not equal to 2. End\n");
      return ErrWavFormat;
   }

   return OK;
}

bool DetectSilenceCPU(MyType, MyType *, MyType *);
/**
  *  \fn    bool DetectSilence(const MyType obsprob, MyType *prob_silen, MyType *prob_audio)
  *  \brief DetectSilence checks whether audio (frame) is silence or audio
  *  \param obsprob:       (in) vaule obtained doing: v_dxStates[1] - v_dxStates[0]
  *  \param prob_silen: (inout) Silence probability
  *  \param prob_audio: (inout) Audio probability
  *  \return: 0 if audio, 1 if silence
*/
bool DetectSilence(MyType obsprob, MyType *prob_silen, MyType *prob_audio)
{
   MyType pss, pas, psa, paa, norm;

   #ifdef SIMPLE
      obsprob = 1 / (1 + expf(-obsprob));
   #else
      obsprob = 1 / (1 + exp(-obsprob));
   #endif

   /* HMM */
   pss = (*prob_silen) * (MyType)HMMremain;
   pas = (*prob_audio) * (MyType)HMMchange;
   psa = (*prob_silen) * (MyType)HMMchange;
   paa = (*prob_audio) * (MyType)HMMremain;

   if (pss > pas)
      (*prob_silen) = pss * obsprob;
   else
      (*prob_silen) = pas * obsprob;

   if (psa > paa)
      (*prob_audio) = psa * (1-obsprob);
   else
      (*prob_audio) = paa * (1-obsprob);

   norm = (*prob_silen) + (*prob_audio);

   (*prob_silen) = (*prob_silen) / norm;
   (*prob_audio) = (*prob_audio) / norm;

   return (*prob_silen > *prob_audio);
}

#endif
