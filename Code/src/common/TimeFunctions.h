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
 *  \file    TimeFunctions.h
 *  \brief   File with functions used by ReMAS, both CPU and GPU, for management the time.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#pragma once

#ifndef ATIME_H
#define ATIME_H

#ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>

int msleep(int);
/**
 *  \fn    int msleep(int milliseconds)
 *  \brief msleep waits milliseconds  
 *  \param milliseconds: (in) time to wait
 *  \return: 0 if all is OK, otherwise a code error (see defines.h)
 *
*/
int msleep(int milliseconds)
{
  /*                       = { seconds,             nanoseconds } */
  struct timespec ts_sleep = { milliseconds / 1000, (milliseconds % 1000) * 1000000L };

  if (nanosleep(&ts_sleep, NULL) < 0) return -11; else return 0;
}

#endif
