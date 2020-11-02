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
 *  \file    MacOSSoundFunctions.h
 *  \brief   File with sound (using CoreAudio) to access to default input audio, both CPU and GPU.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    March 23, 2018
 */

#pragma once

#ifdef CAUDIO
#ifndef MacOSSoundFunctions_h
#define MacOSSoundFunctions_h

#include <AudioToolbox/AudioToolbox.h>
#include "defines.h"
#include "ctype.h"
#include <unistd.h>


#define _POSIX_SOURCE
#include <signal.h>
#include <sys/time.h>

#pragma mark user data struct
#pragma mark user info struct
typedef struct{
    Boolean        running;         //Boolean to keep track of whether the queue is running
    AudioQueueBufferRef buffer;     //Buffer containing the last frame read (Async)
    short *middleBuffer;            //Buffer that contain the 10 las frames readed
    int idReadMB;                   //ID of last frame readed in middleBuffer
    short idWriteMB;                //ID of las frame writed in middleBuffer
    short numLastRead;              //Number of shorts readed in idWriteMB
    short toursReaded;              //Number of reading tours in te middlebuffer circular buffer
    short toursWrited;              //Number of writing tours in te middlebuffer circular buffer
    bool bufferChange;              //Bollean that identify if the buffer read a new frame
} RecorderStruct;

void timer_handler (int signum);
/**
 *  \fn    timer_handler (int signum)
 *  \brief Handler of alarm signal generated for others methods of MacOSSoundFunctions. Never use/call it outside this file.
 *  \param signum: the generated signal integer
 *  \return: void
 */
void timer_handler (int signum)
{
    //NOT implemented method
}

void timer_launch();
/**
 *  \fn    timer_launch()
 *  \brief Laucher of alarm signal generated for others methods of MacOSSoundFunctions, Never use/call it outside this file.
 *  \return: void
 */
void timer_launch(){
    struct sigaction sa;
    
    sa.sa_handler = timer_handler;
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGALRM, &sa, 0);
}

#pragma mark record callback function
static void AQInputCallback(void *inUserData, AudioQueueRef inQueue, AudioQueueBufferRef inBuffer, const AudioTimeStamp *inStartTime, UInt32 inNumPackets, const AudioStreamPacketDescription *inPacketDesc);
/**
 *  \fn    void AQInputCallback(void *inUserData, AudioQueueRef inQueue, AudioQueueBufferRef inBuffer, const AudioTimeStamp *inStartTime, UInt32 inNumPackets, const AudioStreamPacketDescription *inPacketDesc)
 *  \brief Callback launched when the buffer is filled
 *  \param inUserData: Input data of user
 *  \param inQeue: Input qeue of sound buffers
 *  \param inBuffer: Input reference of buffer
 *  \param inStartTime: Input TimeStamp of captured sound
 *  \param inNumPackets: Input number of Packets
 *  \param inPacketDesc: Input description of audio packets
 *  \return: void
 */
static void AQInputCallback(void *inUserData, AudioQueueRef inQueue, AudioQueueBufferRef inBuffer, const AudioTimeStamp *inStartTime, UInt32 inNumPackets, const AudioStreamPacketDescription *inPacketDesc)
{
    int err = 0;
    RecorderStruct *recorder = (RecorderStruct *)inUserData;
    
    if (inNumPackets > 0) {
        recorder->buffer = inBuffer;
        short *goodBuffer = inBuffer->mAudioData;
        
        //Actualize recorder variable
        recorder->idWriteMB = (recorder->idWriteMB + recorder->numLastRead) % MiddleBufferSize;
        if(recorder->idWriteMB == 0) {
            recorder->toursWrited++;
        }
        
        //Compute the middlebuffer to determine if it's filled or no and copy in it the recibed sound
        if(recorder->idWriteMB + (inBuffer->mAudioDataByteSize/sizeof(short)) > MiddleBufferSize-1)
        {
            int toTT = MiddleBufferSize - recorder->idWriteMB;
            while((recorder->idWriteMB > (recorder->idReadMB + MiddleBufferSize*0.5) && recorder->toursReaded == recorder->toursWrited) || (recorder->toursWrited > recorder->toursReaded && (recorder->idWriteMB + MiddleBufferSize*0.5) > recorder->idReadMB) || (recorder->toursWrited < recorder->toursReaded && MiddleBufferSize*0.5 <= recorder->idWriteMB - recorder->idReadMB)){
                timer_launch();
            }
            memcpy(&recorder->middleBuffer[recorder->idWriteMB],goodBuffer, toTT*sizeof(short));
            recorder->numLastRead = (inBuffer->mAudioDataByteSize/sizeof(short)) - toTT;
            recorder->idWriteMB = 0;
            recorder->toursWrited++;
            memcpy(recorder->middleBuffer,&goodBuffer[toTT], recorder->numLastRead*sizeof(short));
        }
        else
        {
            while((recorder->idWriteMB > (recorder->idReadMB + MiddleBufferSize*0.5) && recorder->toursReaded == recorder->toursWrited) || (recorder->toursWrited > recorder->toursReaded && (recorder->idWriteMB + MiddleBufferSize*0.5) > recorder->idReadMB) || (recorder->toursWrited < recorder->toursReaded && MiddleBufferSize*0.5 <= recorder->idWriteMB - recorder->idReadMB)){
                timer_launch();
            }
            memcpy(&recorder->middleBuffer[recorder->idWriteMB],goodBuffer, inBuffer->mAudioDataByteSize);
            recorder->numLastRead = inBuffer->mAudioDataByteSize/sizeof(short);
        }
    }
    if (recorder->running)
    {
        err = AudioQueueEnqueueBuffer(inQueue,inBuffer, 0, NULL);
        recorder->bufferChange = true;
    }
    if (err != noErr)
    {
        fprintf(stderr, "Error: %s (%s)\n", "AQInputCallback failed", "(Enqueue or Write Output file?)");
        exit(1);
    }
}

#pragma mark utility functions
void CheckError(OSStatus error, const char *operation);
/**
 *  \fn    void CheckError(OSStatus error, const char *operation)
 *  \brief CheckError checks if an error occure with the functions that use CoreAudio tools
 *  \param error: contains de OSStatus (normal use -> returns of all functions)
 *  \param operation: Contains a char* with the error return statement
 *  \return: OSStatus: error status
 *
 */
void CheckError(OSStatus error, const char *operation)
{
    if (error == noErr) return;
    
    char errorString[20];
    
    *(UInt32 *)(errorString + 1) = CFSwapInt32HostToBig(error);
    if (isprint(errorString[1]) && isprint(errorString[2]) && isprint(errorString[3]) && isprint(errorString[4]))
    {
        errorString[0] = errorString[5] = '\'';
        errorString[6] = '\0';
    }
    else
    {
        sprintf(errorString, "%d", (int)error);
    }
    
    fprintf(stderr, "Error: %s (%s)\n", operation, errorString);
    exit(1);
}


static int ComputeRecordBufferSize( const AudioStreamBasicDescription *format, AudioQueueRef queue, float frames);
/**
 *  \fn    int ComputeRecordBufferSize( const AudioStreamBasicDescription *format, AudioQueueRef queue, float seconds);
 *  \brief Compute the size of the input buffer of sound
 *  \param format: Description of the audio stream
 *  \param qeue: Qeue of sound buffers
 *  \param seconds: record seconds of input buffer
 *  \return: The input sound device buffer size
 */
static int ComputeRecordBufferSize( const AudioStreamBasicDescription *format, AudioQueueRef queue, float frames)
{
    int packets, bytes;
    if (format->mBytesPerFrame > 0)
        bytes = frames * format->mBytesPerFrame;
    else
    {
        UInt32 maxPacketSize;
        if (format->mBytesPerPacket > 0)
            // Constant packet size
            maxPacketSize = format->mBytesPerPacket;
        else
        {
            // Get the largest single packet size possible
            UInt32 propertySize = sizeof(maxPacketSize);
            CheckError(AudioQueueGetProperty(queue, kAudioConverterPropertyMaximumOutputPacketSize,
                    &maxPacketSize, &propertySize), "Couldn't get queue's maximum output packet size");
        }
        if (format->mFramesPerPacket > 0)
            packets = frames / format->mFramesPerPacket;
        else
            // Worst-case scenario: 1 frame in a packet
            packets = frames;
        // Sanity check
        if (packets == 0)
            packets = 1;
        bytes = packets * maxPacketSize;
    }
    return bytes;
}

void ConfigureAndAllocAudioQueues(RecorderStruct* recorder, AudioQueueRef* queue);
/**
 * \fn   int ConfigureAndAllocAudioQueues(RecorderStruct* recorder, AudioQueueRef* queue)
 * \brief   This function configure and allocate all data needed for audio queues
 * \param   recorder: (out) A pointer of RecorderStruct that all resulting data are vinculed for it
 * \param   queue: the audio queue used
 * \return: void
 */
void ConfigureAndAllocAudioQueues(RecorderStruct* recorder, AudioQueueRef* queue)
{
    //Create and configure the Audio Stream
    AudioStreamBasicDescription recordFormat;
    memset(&recordFormat, 0, sizeof(recordFormat));
    recordFormat.mFormatID |= kAudioFormatLinearPCM;
    
    recordFormat.mFormatFlags &= ~kLinearPCMFormatFlagsAreAllClear;
    recordFormat.mFormatFlags &= ~kLinearPCMFormatFlagIsFloat;
    recordFormat.mFormatFlags |= kLinearPCMFormatFlagIsSignedInteger;
    recordFormat.mFormatFlags &= ~kLinearPCMFormatFlagIsBigEndian;
    recordFormat.mFormatFlags &= ~kLinearPCMFormatFlagIsNonInterleaved;
    recordFormat.mFormatFlags &= ~kLinearPCMFormatFlagIsNonMixable;
    recordFormat.mFormatFlags |= kLinearPCMFormatFlagIsPacked;
    
    recordFormat.mChannelsPerFrame = AQChannels;
    recordFormat.mBitsPerChannel = AQBitsPerChannel;
    recordFormat.mSampleRate = AQRate;
    
    int bytes = (recordFormat.mBitsPerChannel / 8) * recordFormat.mChannelsPerFrame;
    recordFormat.mBytesPerFrame = bytes;
    recordFormat.mBytesPerPacket = bytes;
    
    UInt32 propSize = sizeof(recordFormat);
    CheckError(AudioFormatGetProperty(kAudioFormatProperty_FormatInfo, 0, NULL, &propSize, &recordFormat), "AudioFormatGetProperty failed");
    
    //Allocate recorder struct info
    recorder->middleBuffer = calloc(MiddleBufferSize, sizeof(short));
    recorder->idReadMB = -1;
    recorder->idWriteMB = 0;
    recorder->numLastRead = 0;
    recorder->toursWrited = 0;
    recorder->toursReaded = 1;
    recorder->bufferChange = false;
    
    //Set up queue
    CheckError(AudioQueueNewInput(&recordFormat, AQInputCallback, recorder, NULL, NULL, 0, queue), "AudioQueueNewInput failed");
    UInt32 size = sizeof(recordFormat);
    CheckError(AudioQueueGetProperty(*queue, kAudioConverterCurrentOutputStreamDescription, &recordFormat, &size), "Couldn't get queue's format");
    
    // Other setup as needed
    int bufferByteSize = ComputeRecordBufferSize(&recordFormat, *queue, AQBufferSize);
    
    int bufferIndex;
    for (bufferIndex = 0; bufferIndex < kNumberRecordBuffers; ++bufferIndex)
    {
        AudioQueueBufferRef buffer;
        CheckError(AudioQueueAllocateBuffer(*queue, bufferByteSize, &buffer), "AudioQueueAllocateBuffer failed");
        CheckError(AudioQueueEnqueueBuffer(*queue, buffer, 0, NULL), "AudioQueueEnqueueBuffer failed");
    }
    CheckError(AudioQueueAllocateBuffer(*queue, bufferByteSize, &recorder->buffer), "AudioQueueAllocateBuffer failed");
    
    printf("Rate after config:   %f \nBitsPerChannel:   %d\nChanels:    %d\nBytes Per Frames and Packets:     %d\n",recordFormat.mSampleRate, recordFormat.mBitsPerChannel, recordFormat.mChannelsPerFrame, recordFormat.mBytesPerPacket);
}

int ReadAudioQueue1st(short *frame, RecorderStruct* recorder, FILE *fpdump);
/**
 *  \fn      int ReadAudioQueue1st(frame, &recorder, FILE *fpdump)
 *  \brief ReadAudioQueue1st reads from microphone the first audio (frame)
 *  \param frame:    (out) Vector to store the audio of the first frame
 *  \param recorder:  (in) A pointer of RecorderStruct that all resulting data are vinculed for it
 *  \param fpdump:   (out) File handle when DUMP is active
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
 */
int ReadAudioQueue1st(short *frame, RecorderStruct* recorder, FILE *fpdump)
{
   int i;
    for (i=0; i<=(TAMTRAMA/TAMMUESTRA) ; i++)
    {
        if (recorder->idReadMB < 0)
        {
            recorder->idReadMB = 0;
            while((recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5) && recorder->toursReaded == recorder->toursWrited) || (recorder->toursWrited < recorder->toursReaded && recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5)) || (recorder->toursWrited > recorder->toursReaded && MiddleBufferSize*0.5 <= recorder->idReadMB - recorder->idWriteMB)){
                timer_launch();
            }
            memcpy(&frame[TAMMUESTRA*i],recorder->middleBuffer,TAMMUESTRA*sizeof(short));
            recorder->bufferChange = false;
        }
        else
        {
            recorder->idReadMB = (recorder->idReadMB + TAMMUESTRA) % MiddleBufferSize;
            if(recorder->idReadMB == 0){
                recorder->toursReaded++;
            }
            while((recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5) && recorder->toursReaded == recorder->toursWrited) || (recorder->toursWrited < recorder->toursReaded && recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5)) || (recorder->toursWrited > recorder->toursReaded && MiddleBufferSize*0.5 <= recorder->idReadMB - recorder->idWriteMB)){
                timer_launch();
            }
            memcpy(&frame[TAMMUESTRA*i],&recorder->middleBuffer[recorder->idReadMB],TAMMUESTRA*sizeof(short));
            recorder->bufferChange = false;
        }
    }

   #ifdef DUMP
      if (fwrite(&frame[TAMMUESTRA], sizeof(short), TTminusTM, fpdump) != TTminusTM) return ErrWriteFile;
   #endif
   
   return OK;
}

int ReadAudioQueue(short *frame, RecorderStruct* recorder, FILE *fpdump);
/**
 *  \fn      int ReadAudioQueue(frame, &recorder, fp)
 *  \brief ReadAudioQueue reads from microphone the audio (frame)
 *  \param frame:    (out) Vector to store the audio of the first frame
 *  \param recorder:  (in) A pointer of RecorderStruct that all resulting data are vinculed for it
 *  \param fpdump:   (out) File handle when DUMP is active
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
 */
int ReadAudioQueue(short *frame, RecorderStruct* recorder, FILE *fpdump)
{
    memmove(frame, &frame[TAMMUESTRA], sizeof(short)*TTminusTM);
    recorder->idReadMB = (recorder->idReadMB + TAMMUESTRA) % MiddleBufferSize;
    
    if(recorder->idReadMB == 0) {
        recorder->toursReaded++;
    }
    while((recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5) && recorder->toursReaded == recorder->toursWrited) || (recorder->toursWrited < recorder->toursReaded && recorder->idWriteMB < (recorder->idReadMB + MiddleBufferSize*0.5)) || (recorder->toursWrited > recorder->toursReaded && MiddleBufferSize*0.5 <= recorder->idReadMB - recorder->idWriteMB)){
        timer_launch();
    }
        memcpy(&frame[TTminusTM],&recorder->middleBuffer[recorder->idReadMB],TAMMUESTRA*sizeof(short));
        recorder->bufferChange = false;
   #ifdef DUMP
     if (fwrite(&frame[TTminusTM], sizeof(short), TAMMUESTRA, fpdump) != TAMMUESTRA) return ErrWriteFile;
   #endif

    
    return OK;
}


#endif /* MacOSSoundFunctions_h */
#endif /* CAUDIO: Core Audio in DARWIN */
