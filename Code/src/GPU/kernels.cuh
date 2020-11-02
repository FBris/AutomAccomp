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
 *  \file    kernels.cuh
 *  \brief   File with ReMAS kernels for Nvidia GPUs
 *  \author  Information Retrieval and Parallel Computing Group, University of Oviedo, Spain
 *  \author  Interdisciplinary Computation and Communication Group, Universitat Politecnica de Valencia, Spain
 *  \author  Signal Processing and Telecommunication Systems Research Group, University of Jaen, Spain.
 *  \author  Contact: remaspack@gmail.com
 *  \date    February 13, 2017
*/

/**
 *  \fn    __device__ inline double __shfl_downD(double var, unsigned int srcLane, int width=32)
 *  \brief __shfl_downD performs __shfl_down of a double number
 *  \param var:    (inout) The data
 *  \param srcLane:   (in) The lane
 *  \param width:     (in) Width of line
 *  \return: The __shfl_down of a double number
*/
__device__ inline double __shfl_downD(double var, unsigned int srcLane, int width=32)
{
  int2 a = *reinterpret_cast<int2*>(&var);

  #ifdef CUDA9
     a.x = __shfl_down_sync(0x0, a.x, srcLane, width);
     a.y = __shfl_down_sync(0x0, a.y, srcLane, width);
  #else
     a.x = __shfl_down(a.x, srcLane, width);
     a.y = __shfl_down(a.y, srcLane, width);
  #endif

  return *reinterpret_cast<double*>(&a);
}


/**
 *  \fn    __inline__ __device__ double warpReduceSumD(double val)
 *  \brief warpReduceSumD does double sum reduction within a warp
 *  \param val: (in) The data
 *  \return: The sum reduction of double type within a warp
*/
__inline__ __device__ double warpReduceSumD(double val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_downD(val, offset);
  return val;
}


/**
 *  \fn    __inline__ __device__ double warpReduceSumS(float val)
 *  \brief warpReduceSumD does float sum reduction within a warp
 *  \param val: (in) The data
 *  \return: The sum reduction of float type within a warp
*/
__inline__ __device__ float warpReduceSumS(float val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    #ifdef CUDA9
       val += __shfl_down_sync(0x0, val, offset, 32);
    #else
       val += __shfl_down(val, offset, 32);
    #endif

  return val;
}


/**
 *  \fn    __global__ void kernel_InitDTW(MyType* __restrict__ pV, const int pos, const int size)
 *  \brief kernel_InitDTW This cuda kernel initializes DTW vector
 *  \param pV:  (out) DTW pV vector
 *  \param pos:  (in) One special position within the vectors
 *  \param size: (in) Size of DTW vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_InitDTW(MyType* __restrict__ pV, const int pos, const int size)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
     
   if (tid < size)
   {
      if (tid==pos)
         pV[tid]=0.0;
      else
        #ifdef SIMPLE
          pV[tid]=FLT_MAX;      
        #else
          pV[tid]=DBL_MAX;
        #endif
   }
}

/**
 *  \fn    __global__ void kernel_DTW(const MyType* __restrict__ Sequence, MyType* __restrict__ pD, const int NSeq, const int Where, const int NST) 
 *  \brief kernel_DTW This cuda kernel performs the Online-DTW process for the current frame
 *  \param Sequence: (in) Currente frame
 *  \param pD:      (out) DTW pD vector
 *  \param NSeq:     (in) One referenced position within DTW vectors
 *  \param Where:    (in) Position within DTW vectors
 *  \param NST:      (in) Number of states
 *  \return: Nothing, it is a cuda kernel
*/
__global__
// __launch_bounds__(512,4)
void kernel_DTW(const MyType* __restrict__ Sequence, MyType* __restrict__ pD, const int NSeq, const int Where, const int NST) 
{
   int j=threadIdx.x + blockIdx.x * blockDim.x;
   int NSTplusNC, k, Pos;

   MyType d, d2;
   
   #ifdef SIMPLE
      d=FLT_MAX;
   #else
      d=DBL_MAX;
   #endif

   if( j<NST )
   {
      NSTplusNC = N_COSTS + NST;
      Pos       =((NSeq + N_COSTS) % TBLOCK) * NSTplusNC + N_COSTS + j - 1;
      for(k=0; k<N_COSTS; k++)
      {
         d2 = Sequence[j]*CCosts[k]+pD[Pos-k];
         if (d2 < d) d=d2;
      }

      for (k=N_COSTS; k<T_COSTS; k++)
      {
         Pos=((NSeq + (T_COSTS-k)) % TBLOCK) * NSTplusNC + N_COSTS + j - 1;
         
         d2 = Sequence[j]*CCosts[k]+pD[Pos];

         if (d2 < d) d=d2;
      }

      pD[Where+j] = d;
   }
}


/** 
 *  \fn    __global__ void kernel_InitSxD(MyType* __restrict__ odata, MyType* __restrict__ v_SxD, const MyType* __restrict__ v_dxState, const int* __restrict__ I_SxD, const int blockSize, const bool SizeIsPow2, const int size)
 *  \brief kernel_InitSxD This cuda kernel sets up the vector SxD.
 *  \param odata:   (inout) Intermedial data vector
 *  \param v_SxD:     (out) v_SxD vector
 *  \param v_dxState: (out) v_dxState vector
 *  \param I_SxD:     (out) I_SxD vector
 *  \param blockSize:  (in) BlockSize used
 *  \param SizeIsPow2: (in) True if the vector size is power of 2
 *  \param size:       (in) Size of the vector
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_InitSxD(MyType* __restrict__ odata, MyType* __restrict__ v_SxD, const MyType* __restrict__ v_dxState,
                               const int* __restrict__ I_SxD, const int blockSize, const bool SizeIsPow2, const int size)
{
   extern __shared__ MyType sdata[];

   unsigned int      tid = threadIdx.x;
   unsigned int        i = blockIdx.x*blockSize*2 + threadIdx.x;
   unsigned int gridSize = blockSize*2*gridDim.x;

   MyType mySum=0.0, myData;

   while (i < size)
   {
      myData = v_SxD[i] = v_dxState[I_SxD[i]]; 
      mySum += myData*myData;

      if (SizeIsPow2 || i + blockSize < size)
      {
         myData = v_SxD[i+blockSize] = v_dxState[I_SxD[i+blockSize]];
         mySum += myData*myData;
      }

      i += gridSize;
   }
   sdata[tid] = mySum;
   __syncthreads();

   if ((blockSize >= 512) && (tid < 256))
      sdata[tid] = mySum = mySum + sdata[tid + 256];
   __syncthreads();

   if ((blockSize >= 256) &&(tid < 128))
      sdata[tid] = mySum = mySum + sdata[tid + 128];
   __syncthreads();

   if ((blockSize >= 128) && (tid <  64))
      sdata[tid] = mySum = mySum + sdata[tid +  64];
   __syncthreads();

   if (tid < 32)
   {
      if (blockSize >=  64)
         mySum += sdata[tid + 32];

      for (int offset = sizeWarp/2; offset > 0; offset /= 2) 
         #ifdef CUDA9
            mySum += __shfl_down_sync(0x0, mySum, offset);
         #else
            mySum += __shfl_down(mySum, offset);
         #endif
   }
   if (tid == 0) odata[blockIdx.x] = mySum;
}


/** 
 *  \fn    __global__ void kernel_Sum(MyType* __restrict__ odata, const MyType* __restrict__ idata, const int blockSize, const bool SizeIsPow2, const int size)
 *  \brief kernel_Sum This cuda kernel adds the elements of a vector.
 *  \param odata:   (inout) Intermedial data vector
 *  \param idata:      (in) vector with values to add
 *  \param blockSize:  (in) BlockSize used
 *  \param SizeIsPow2: (in) True if the vector size is power of 2
 *  \param size:       (in) Size of the vector
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_Sum(MyType* __restrict__ odata, const MyType* __restrict__ idata,
                           const int blockSize, const bool SizeIsPow2, const int size)
{
   extern __shared__ MyType sdata[];

   unsigned int      tid = threadIdx.x;
   unsigned int        i = blockIdx.x*blockSize*2 + threadIdx.x;
   unsigned int gridSize = blockSize*2*gridDim.x;

   MyType mySum=0.0;

   while (i < size)
   {
      mySum += idata[i];

      if (SizeIsPow2 || i + blockSize < size)
         mySum += idata[i+blockSize];

      i += gridSize;
   }
   sdata[tid] = mySum;
   __syncthreads();

   if ((blockSize >= 512) && (tid < 256))
      sdata[tid] = mySum = mySum + sdata[tid + 256];
   __syncthreads();

   if ((blockSize >= 256) &&(tid < 128))
      sdata[tid] = mySum = mySum + sdata[tid + 128];
   __syncthreads();

   if ((blockSize >= 128) && (tid <  64))
      sdata[tid] = mySum = mySum + sdata[tid +  64];
   __syncthreads();

   if (tid < 32)
   {
      if (blockSize >=  64)
         mySum += sdata[tid + 32];

      for (int offset = sizeWarp/2; offset > 0; offset /= 2)
         #ifdef CUDA9
            mySum += __shfl_down_sync(0x0, mySum, offset);
         #else
            mySum += __shfl_down(mySum, offset);
         #endif
   }
   if (tid == 0) odata[blockIdx.x] = mySum;
}


/** 
 *  \fn    __global__ void kernel_Vnorm(MyType* __restrict__ odata)
 *  \brief kernel_Vnorm This cuda kernel initializes position 0 of a vector
 *  \param odata: (inout) The vector
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_Vnorm(MyType* __restrict__ odata)
{
   #ifdef SIMPLE
      odata[0] = 1.0f / (sqrtf(odata[0]) + FLT_EPSILON);
   #else
      odata[0] = 1.0  / ( sqrt(odata[0]) + DBL_EPSILON);
   #endif
}

/** 
 *  \fn    kernel_ApplyWindow(MyType* __restrict__ X_fft, const short* __restrict__ frame, const MyType* __restrict__ v_hanning, const int TTRA, const int NFFT)
 *  \brief kernel_ApplyWindow scales and set the elements of the audio vector X_fft
 *  \param X_fft:     (out) The vector to scale and set
 *  \param frame:      (in) The vector with current audio (frame)
 *  \param v_hanning:  (in) The hanning vector
 *  \param TTRA:       (in) Size of the frame
 *  \param NFFT:       (in) Size of the X_fft vector
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_ApplyWindow(MyType* __restrict__ X_fft, const short* __restrict__ frame,
                                   const MyType* __restrict__ v_hanning, const int TTRA, const int NFFT)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < NFFT)
      X_fft[tid] = (tid < TTRA) ? (MyType)frame[tid] * Scaling * v_hanning[tid] : 0.0;
}


/** 
 *  \fn    __global__ void kernel_UpdateSxD(MyType* __restrict__ dest, const MyType ALPHA, const MyType* __restrict__ norm, const int size)
 *  \brief kernel_UpdateSxD This cuda kernel update the elements of SxD vector
 *  \param dest: (inout) The vector SxD
 *  \param ALPHA:   (in) The value of parameter ALPHA
 *  \param norm:    (in) Vector with the norm in position 0
 *  \param size:    (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_UpdateSxD(MyType* __restrict__ dest, const MyType ALPHA, const MyType* __restrict__ norm,
                                 const int size)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if (tid < size)
      #ifdef SIMPLE
         dest[tid] = 1.0f - expf(ALPHA*fabsf(dest[tid]*norm[0]));
      #else
         dest[tid] = 1.0  -  exp(ALPHA* fabs(dest[tid]*norm[0]));
     #endif
}


/** 
 *  \fn    __global__ void kernel_CompNorB0(MyType* __restrict__ norms, const MyType value, const int size)
 *  \brief kernel_CompNorB0 This cuda kernel computes the norm of a vector when BETA=0
 *  \param norms: (out) The output vector
 *  \param value:  (in) The value for the initialization of the vector
 *  \param size:   (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_CompNorB0(MyType* __restrict__ norms, const MyType value, const int size)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (tid < size)
      norms[tid]=value;
}


/** 
 *  \fn    __global__ void kernel_CompNorB1(MyType* __restrict__ norms, const MyType* __restrict__ s_fk, const int NMIDI, const int size)
 *  \brief kernel_CompNorB1 This cuda kernel computes the norm of a vector when BETA=1
 *  \param norms: (out) The output vector
 *  \param s_fk:   (in) Vector s_fk
 *  \param NMIDI:  (in) Number of midi notes
 *  \param size:   (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_CompNorB1(MyType* __restrict__ norms, const MyType* __restrict__ s_fk,
                                 const int NMIDI, const int size)
{
   unsigned int i = blockIdx.x * blockDim.y + threadIdx.y;
   unsigned int j;
   unsigned int stride = i*N_MIDI_PAD;
   MyType a;

   if (i<size)
   {
     a=0.0; 
     for(j=threadIdx.x; j<NMIDI; j+=32)
        a += s_fk[stride+j];

     #ifdef SIMPLE
       a = warpReduceSumS(a);
     #else
       a = warpReduceSumD(a);
     #endif

     if (threadIdx.x==0) norms[i]=a;
   }
}


/** 
 *  \fn    __global__ void kernel_CompNorBG(MyType* __restrict__ norms, MyType* __restrict__ ts_fk, const MyType* __restrict__ s_fk, const int NMIDI, const MyType BETA, const int size)
 *  \brief kernel_CompNorBG This cuda kernel computes the norm of a vector when BETA <> 0 and BETA <> 1
 *  \param norms: (out) The output vector
 *  \param ts_fk: (out) Vector ts_fk
 *  \param s_fk:   (in) Vector s_fk
 *  \param NMIDI:  (in) Number of midi notes
 *  \param BETA:   (in) Value of parameter BETA
 *  \param size:   (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_CompNorBG(MyType* __restrict__ norms, MyType* __restrict__ ts_fk,
                                 const MyType* __restrict__ s_fk, const int NMIDI, const MyType BETA, const int size)
{
   unsigned int i = blockIdx.x * blockDim.y + threadIdx.y;
   unsigned int j;
   unsigned int stride = i*N_MIDI_PAD;
   MyType a,b;

   if (i<size)
   {
     #ifdef SIMPLE
       a=0.0f; 
       for(j=threadIdx.x; j<NMIDI; j+=32)
       {
         ts_fk[stride+j] = b = powf(s_fk[stride+j], BETA - 1.0f);
         a += b*s_fk[stride+j];
       }
       a = warpReduceSumS(a);
     #else
       a = 0.0; 
       for(j=threadIdx.x; j<NMIDI; j+=32)
       {
         ts_fk[stride+j] = b = pow(s_fk[stride+j], BETA - 1.0f);
         a += b*s_fk[stride+j];
       }
       a = warpReduceSumD(a);
     #endif

     if (threadIdx.x==0) norms[i]=a;
   }
}

/** 
 *  \fn    __global__ void kernel_PowToReal(MyType* __restrict__ dest, const MyType* __restrict__ src, const MyType ex, const int size)
 *  \brief kernel_PowToReal This cuda kernel powers the elements of a vector to a real number and stores them in other vector
 *  \param dest: (out) The output vector
 *  \param src:   (in) The input vector
 *  \param ex:    (in) Number of midi notes
 *  \param size:  (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_PowToReal(MyType* __restrict__ dest, const MyType* __restrict__ src, const MyType ex, const int size)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < size)
   {
      #ifdef SIMPLE
        dest[tid]=powf(src[tid], ex);
      #else
        dest[tid]= pow(src[tid], ex);
      #endif
   }
}


/** 
 *  \fn    __global__ void kernel_Modul(MyType* __restrict__ dest, const MyType* __restrict__ src, const int size)
 *  \brief kernel_Modul This cuda kernel computes the modulus of elements of a vector and stores them in other vector
 *  \param dest: (out) The output vector
 *  \param src:   (in) The input vector
 *  \param size:  (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_Modul(MyType* __restrict__ dest, const MyType* __restrict__ src, const int size)
{
  unsigned int    tid =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = (threadIdx.x + blockIdx.x * blockDim.x)*2;
  
  MyType tmp1, tmp2;

  if (tid <= size)
  {
     tmp1 = src[stride];
     tmp2 = src[stride + 1];
     
     dest[tid]=tmp1*tmp1 + tmp2*tmp2;
  }
}


/** 
 *  \fn    __global__ void kernel_Cfreq(MyType* __restrict__ dest, const MyType* __restrict__ src)
 *  \brief kernel_Cfreq This cuda kernel computes sqrt(sum of elements of a vector) and stores it in dest[0]
 *  \param dest: (out) The output vector
 *  \param src:   (in) The input vector
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_Cfreq(MyType* __restrict__ dest, const MyType* __restrict__ src)
{
  unsigned int i = blockIdx.x;
  unsigned int j = threadIdx.x;

  MyType tmp = 0.0;
  for( unsigned int k=Ckmin_fft[i]+j; k<=Ckmax_fft[i]; k+=32 ) {
    tmp += src[k];
  }

  #ifdef SIMPLE
     tmp = warpReduceSumS(tmp);
  #else
     tmp = warpReduceSumD(tmp);
  #endif

  if( j==0 ) {
    #ifdef SIMPLE
       dest[i] = sqrtf(tmp);
    #else
       dest[i] =  sqrt(tmp);
    #endif
  }
}


/** 
 *  \fn    __global__ void kernel_Reduction(MyType* __restrict__ dest, const int size)
 *  \brief kernel_Reduction This cuda kernel performs a typical sum-reduction of a vector
 *  \param dest: (inout) The vector
 *  \param size:    (in) The vector size
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void  kernel_Reduction(MyType* __restrict__ dest, const int size)
{
   unsigned int tid = threadIdx.x;
   unsigned int j;

   MyType a=0.0;

   for(j=tid; j<size; j+=32) a += dest[j];

   #ifdef SIMPLE
     a = warpReduceSumS(a);
   #else
     a = warpReduceSumD(a);
   #endif

   if (tid==0) dest[size]=a;
}

/** 
 *  \fn    __global__ void kernel_ReductionPowBeta(MyType* __restrict__ dest, const MyType BETA, const int size)
 *  \brief kernel_Reduction This cuda kernel performs a typical sum-reduction of a vector
 *  \param dest: (inout) The vector
 *  \param BETA:    (in) The value of BETA
 *  \param size:    (in) The vector size
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void  kernel_ReductionPowBeta(MyType* __restrict__ dest, const MyType BETA, const int size)
{
   unsigned int tid = threadIdx.x;
   unsigned int j;

   MyType a=0.0;

   for(j=tid; j<size; j+=32)
     #ifdef SIMPLE
       a += powf(dest[j], BETA);
     #else
       a += pow (dest[j], BETA);
     #endif

   #ifdef SIMPLE
     a = warpReduceSumS(a);
     if (tid==0) dest[size]=powf(a, 1.0/BETA);
   #else
     a = warpReduceSumD(a);
     if (tid==0) dest[size]=pow(a, 1.0/BETA);
   #endif
}


/* Paralelizacion bloques de 32x32 / 16*32  y sufle registros */
/** 
 *  \fn    __global__ void __launch_bounds__(512,4) kernel_CompDisB0(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq, const MyType* __restrict__ norms, const MyType* __restrict__ s_fk, const int NMIDI, const int size)
 *  \brief kernel_CompDisB0 This cuda kernel computes the distortion of a vector when BETA=0
 *  \param dest:   (out) The output vector
 *  \param v_cfreq: (in) Vector v_cfreq
 *  \param norms:   (in) Vector norms
 *  \param s_fk:    (in) Vector s_fk
 *  \param NMIDI:   (in) Number of midi notes
 *  \param size:    (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void __launch_bounds__(512,4)
kernel_CompDisB0(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq, const MyType* __restrict__ norms,
                 const MyType* __restrict__ s_fk, const int NMIDI, const int size)
{
   unsigned int      i =  blockIdx.x * blockDim.y + threadIdx.y;
   unsigned int      j;
   unsigned int stride = i * N_MIDI_PAD;
   unsigned int th_row = threadIdx.y;
   unsigned int th_col = threadIdx.x;
   unsigned int    row = i + threadIdx.x; /* This is useful only for the first row */
   bool          guard = th_row == 0 && row < size && th_col < blockDim.y;
   MyType a, b, tmp1;

   __shared__ MyType sh[32];

   if (i < size)
   {
      a = 0.0;
      for( j=th_col; j<NMIDI; j+=32) {
        a += v_cfreq[j] / s_fk[stride+j];
      }
      #ifdef SIMPLE
         a = warpReduceSumS(a);
      #else
         a = warpReduceSumD(a);
      #endif
      if( guard ) {
        sh[th_col] = norms[row];
      }
      __syncthreads();
      if (th_col == 0) { b = a / sh[th_row]; }
         #ifdef CUDA9
            b = __shfl_sync(0x0, b, 0);
         #else
            b = __shfl(b, 0);
         #endif

      a = 0.0;
      for( j=th_col; j<NMIDI; j+=32) 
      {
         tmp1 = v_cfreq[j] / (s_fk[stride + j] * b);
         #ifdef SIMPLE
            a += tmp1 - logf(tmp1) - 1.0f;
         #else
            a += tmp1 -  log(tmp1) - 1.0;
         #endif
      }                                                          
      #ifdef SIMPLE
         a = warpReduceSumS(a);
      #else
         a = warpReduceSumD(a);
      #endif
      if( th_col == 0 ) {
        sh[th_row] = a;
      }
      __syncthreads();
      if( guard ) {
        dest[row] = sh[th_col];
      }
   }
}


/** 
 *  \fn    __global__ void __launch_bounds__(512, 4) kernel_CompDisB1(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq, const MyType* __restrict__ norms, const MyType* __restrict__ s_fk, const int NMIDI, const int size)
 *  \brief kernel_CompDisB1 This cuda kernel computes the distortion of a vector when BETA=1
 *  \param dest:   (out) The output vector
 *  \param v_cfreq: (in) Vector v_cfreq
 *  \param norms:   (in) Vector norms
 *  \param s_fk:    (in) Vector s_fk
 *  \param NMIDI:   (in) Number of midi notes
 *  \param size:    (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void __launch_bounds__(512, 4)
kernel_CompDisB1(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq, const MyType* __restrict__ norms,
                 const MyType* __restrict__ s_fk, const int NMIDI, const int size)
{
   unsigned int      i =  blockIdx.x * blockDim.y + threadIdx.y;
   unsigned int      j;
   unsigned int stride = i * N_MIDI_PAD;
   unsigned int th_row = threadIdx.y;
   unsigned int th_col = threadIdx.x;
   unsigned int    row = i + threadIdx.x; /* This is useful only for the first row */
   bool          guard = th_row == 0 && row < size && th_col < blockDim.y;
   MyType a, tmp1, tmp2, tmp3;

   __shared__ MyType sh[32];

   if ( i < size )
   {
      if( guard ) {
        sh[th_col] = v_cfreq[NMIDI] / norms[row];
      }
      __syncthreads();
      tmp1 = sh[th_row];

      a = 0.0;
      for( j=th_col; j<NMIDI; j+=32 ) {
        tmp2 = s_fk[stride+j] * tmp1; 
        tmp3 = v_cfreq[j];
        #ifdef SIMPLE
          a += tmp3*logf(tmp3/tmp2) + tmp2 - tmp3;
        #else
          a += tmp3* log(tmp3/tmp2) + tmp2 - tmp3;
        #endif 
      }
      #ifdef SIMPLE
         a = warpReduceSumS(a);
      #else
         a = warpReduceSumD(a);
      #endif
      if( th_col == 0 ) {
        sh[th_row] = a; 
      }
      __syncthreads();
      if( guard ) {
        dest[row] = sh[th_col];
      }
   }
}


/* Paralelizacion bloques de 32x32 / 16*32  y sufle registros */
/** 
 *  \fn    __global__ void __launch_bounds__(512, 4) kernel_CompDisBG(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq, const MyType* __restrict__ norms, const MyType* __restrict__ s_fk, const MyType* __restrict__ ts_fk, const MyType* __restrict__ tauxi, const MyType BETA, const int NMIDI, const int size)
 *  \brief kernel_CompDisBG This cuda kernel computes the distortion of a vector when BETA <> 0 and BETA <> 1
 *  \param dest:   (out) The output vector
 *  \param v_cfreq: (in) Vector v_cfreq
 *  \param norms:   (in) Vector norms
 *  \param s_fk:    (in) Vector s_fk
 *  \param ts_fk:   (in) Vector ts_fk
 *  \param tauxi:   (in) Vector tauxi
 *  \param BETA:    (in) BETA value
 *  \param NMIDI:   (in) Number of midi notes
 *  \param size:    (in) Size of the vectors
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void __launch_bounds__(512, 4)
kernel_CompDisBG(MyType* __restrict__ dest, const MyType* __restrict__ v_cfreq,
                 const MyType* __restrict__ norms, const MyType* __restrict__ s_fk,
                 const MyType* __restrict__ ts_fk, const MyType* __restrict__ tauxi,
                 const MyType BETA, const int NMIDI, const int size)
{
   unsigned int      i =  blockIdx.x * blockDim.y + threadIdx.y;
   unsigned int      j, k;
   unsigned int stride = i * N_MIDI_PAD;
   unsigned int th_row = threadIdx.y;
   unsigned int th_col = threadIdx.x;
   unsigned int    row = i + threadIdx.x; /* This is useful only for the first row */
   bool          guard = th_row == 0 && row < size && th_col < blockDim.y;
   MyType a, b, tmp1, tmp2, beta1 = BETA-1.0, tmp3 = (1.0 / (BETA * beta1));

   __shared__ MyType sh_a[16], sh_b[16];

   if ( i < size )
   {
      a = 0.0;
      for(j=th_col,k=stride+th_col;j<NMIDI;j+=32,k+=32) {
        a += v_cfreq[j] * ts_fk[stride+j];
      }
      #ifdef SIMPLE
      a = warpReduceSumS(a);
      #else
      a = warpReduceSumD(a);
      #endif
      if( th_col == 0 ) {
        sh_a[th_row] = a;
      }
      __syncthreads();
      if( guard ) {
        a = sh_a[th_col] / norms[row];
        #ifdef SIMPLE
        b = powf(a, beta1);
        #else
        b =  pow(a, beta1);
        #endif
        sh_b[th_col] = BETA * b;
        sh_a[th_col] = b * a * beta1;
      }
      __syncthreads();
      tmp1 = sh_b[th_row];
      tmp2 = sh_a[th_row];
      j = th_col; k = stride+th_col;
      a  = ((tauxi[j] + ts_fk[k] * (s_fk[k] * tmp2 - v_cfreq[j] * tmp1)) * tmp3);
      j += 32; k += 32;
      a += ((tauxi[j] + ts_fk[k] * (s_fk[k] * tmp2 - v_cfreq[j] * tmp1)) * tmp3);
      j += 32; k += 32;
      a += ((tauxi[j] + ts_fk[k] * (s_fk[k] * tmp2 - v_cfreq[j] * tmp1)) * tmp3);
      j += 32; k += 32;
      if( th_col<18 ) {
        a += ((tauxi[j] + ts_fk[k] * (s_fk[k] * tmp2 - v_cfreq[j] * tmp1)) * tmp3);
      }
      #ifdef SIMPLE
      a = warpReduceSumS(a);
      #else
      a = warpReduceSumD(a);
      #endif
      if( th_col == 0 ) {
        sh_a[th_row] = a;
      }
      __syncthreads();
      if( guard ) {
        dest[row] = sh_a[th_col];
      }
   }
}


/** 
 *  \fn    __global__ void kernel_Shift(short* __restrict__ frame, const int TTRAMA, const int TMUEST)
 *  \brief kernel_Shift shifts the vector elements TMUEST positions on the left
 *  \param frame: (out) The vector
 *  \param TTRAMA: (in) Number of elements of the TRAMA
 *  \param TMUEST: (in) Number of elements of the MUESTA
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_Shift(short* __restrict__ frame, const int TTRAMA, const int TMUEST)
{
   unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int i, tmp;
   
   for (i=0; i<(TTRAMA/TMUEST - 1); i++)
   {
      tmp=tid+i*TMUEST;
      frame[tmp]=frame[tmp+TMUEST];
      __syncthreads();
   }
}


/** 
 *  \fn    __global__ void kernel_BetaNorm(MyType* __restrict__ vector, const int size)
 *  \brief kernel__BetaNorm normalized the vector
 *  \param vector: (out) The vector
 *  \param size:    (in) The vector size
 *  \return: Nothing, it is a cuda kernel
*/
__global__ void kernel_BetaNorm(MyType* __restrict__ vector, const int size)
{
   unsigned int tid = threadIdx.x;

   /* The previous call to kernel_Reduction / kernel_ReductionPowBeta puts in vector[size] the reduction value */
   MyType value=vector[size];

   vector[tid] = vector[tid] / value;
}
