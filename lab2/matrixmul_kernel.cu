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

  int by = blockIdx.y;
  int bx = blockIdx.x;

  int ty = threadIdx.y;
  int tx = threadIdx.x;


  float sub_sum = 0;

  int cur_row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int cur_col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

  int end_block = (BLOCK_SIZE + M.width - 1)/BLOCK_SIZE;
  for (int m = 0; m < end_block; ++m) {

    int offset = m*BLOCK_SIZE + tx;

    if ( offset < M.width && cur_row < M.height)
    {
      shared_A[ty][tx] = M.elements[cur_row*M.width + offset];
    }
    else
    {
      shared_A[ty][tx] = 0.0;
    }

    offset = m*BLOCK_SIZE + ty;
    if ( offset < N.height && cur_col < N.width)
    {
      shared_B[ty][tx] = N.elements[offset*N.width + cur_col];
    }
    else
    {
      shared_B[ty][tx] = 0.0;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) sub_sum += shared_A[ty][i] * shared_B[i][tx];

    __syncthreads();
  }

  if (cur_row < P.height && cur_col < P.width)
  {
    P.elements[((by * blockDim.y + ty)*P.width)+(bx*blockDim.x)+tx]=sub_sum;
  }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
