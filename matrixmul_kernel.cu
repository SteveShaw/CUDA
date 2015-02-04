/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

#define BLOCK_SIZE 16
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  int b_row = blockIdx.y;
  int b_col = blockIdx.x;


  //Get sub-matrix
  Matrix sub_P;
  sub_P.width = BLOCK_SIZE;
  sub_P.height = BLOCK_SIZE;
  sub_P.pitch = P.pitch;
  sub_P.elements = &P.elements[P.pitch*BLOCK_SIZE*b_row+BLOCK_SIZE*b_col];

  //accmluate for this sub matrix
  float sum_sub_P = 0.0;

  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  Matrix sub_M;
  Matrix sub_N;

  sub_M.width = M.width;
  sub_M.height = M.height;
  sub_M.pitch = M.pitch;

  sub_N.width = N.width;
  sub_N.height = N.height;
  sub_N.pitch = N.pitch;


  for(int m = 0;m<(M.width/BLOCK_SIZE);++m)
  {
    //Get sub M
    sub_M.elements = &M.elements[M.pitch*BLOCK_SIZE*b_row+BLOCK_SIZE*m];
    //Get sub N
    sub_N.elements = &N.elements[N.pitch*BLOCK_SIZE*m+BLOCK_SIZE*b_col];

    //Read data to shared memory
    __shared__ float sub_M_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sub_N_shared[BLOCK_SIZE][BLOCK_SIZE];

    sub_M_shared[thread_row][thread_col] = M.elements[thread_row*M.pitch+thread_col];
    sub_N_shared[thread_row][thread_col] = N.elements[thread_row*M.pitch+thread_col];

    __syncthreads(); //make sure data are loaded

    for(int i = 0;i<BLOCK_SIZE;++i)
    {
      sum_sub_P += sub_M_shared[thread_row][i]*sub_N_shared[i][thread_col];
    }

    __syncthreads();//wait for computing is finished

    //write back result
    sub_P.elements[thread_row*sub_P.pitch+thread_col] = sum_sub_P;
  }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
