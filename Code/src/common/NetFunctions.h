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
 *  \file    NetFunctions.h
 *  \brief   File with functions used by ReMAS, both CPU and GPU, for management communications.
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/
#pragma once

#ifndef ANET_H
#define ANET_H

#include "defines.h"

#ifdef OSC
  #include <lo.h>
  int SendTempo(lo_address t, int tempo);
  int  SendPlay(lo_address t);
  
  /**
   *  \fn    int SendTempo(lo_address t, int tempo)
   *  \brief SendTempo sends an OSC message with the new tempo to synthesizer 
   *  \param t:     (in) address where the OSC message is sent
   *  \param tempo: (in) tempo value
   *  \return: 0 if all is OK, otherwise a code error (see defines.h)
   *
  */
  int SendTempo(lo_address t, int tempo)
  { 
    if (lo_send(t, "/tempo", "i", tempo) < 0) return ErrSendOSC; else return OK;
  }


  /**
   *  \fn    int SendPlay(lo_address t)
   *  \brief SendPlay sends an OSC message to synthesizer to play or stop the score
   *  \param t: (in) address where the OSC message is sent
   *  \return: 0 if all is OK, otherwise a code error (see defines.h)
   *
  */
  int SendPlay(lo_address t)
  {
    if (lo_send(t, "/actions/play", "b", "\0") < 0) return ErrSendOSC; else return OK;
  }
#endif

#endif
