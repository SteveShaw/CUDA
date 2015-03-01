#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

static unsigned int* d_Data = NULL;
static unsigned int* d_Histogram = NULL;

__global__ void computeHistogram(unsigned int  *buffer, int size, unsigned int *histo )
{
   __shared__ unsigned int temp[1024];

    temp[threadIdx.x + 0] = 0;
    temp[threadIdx.x + 256] = 0;
    temp[threadIdx.x + 512] = 0;
    temp[threadIdx.x + 768] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size)
    {
             atomicAdd( &temp[buffer[i]], 1);
             i += offset;
    }
    __syncthreads();


   atomicAdd( &(histo[threadIdx.x + 0]), temp[threadIdx.x + 0] );
   atomicAdd( &(histo[threadIdx.x + 256]), temp[threadIdx.x + 256] );
   atomicAdd( &(histo[threadIdx.x + 512]), temp[threadIdx.x + 512] );
   atomicAdd( &(histo[threadIdx.x + 768]), temp[threadIdx.x + 768] );

}

void opt_init(unsigned int** h_Data, int width, int height)
{
  cudaMalloc((void **)&d_Histogram, HISTO_HEIGHT * HISTO_WIDTH * sizeof(unsigned int));
  cudaMemset( d_Histogram, 0,HISTO_HEIGHT * HISTO_WIDTH * sizeof( unsigned int ));

    cudaMalloc((void **)&d_Data, width*height*sizeof(unsigned int));
    for(int i = 0;i<height;++i)
    {
      unsigned int *cur = h_Data[i];
      cudaMemcpy(d_Data+i*height, cur, width*sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
}

void opt_2dhisto(int size)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
  computeHistogram<<<2*8,1024, 1024*sizeof(unsigned int)>>>( d_Data,size,d_Histogram );
}

void opt_free()
{
  cudaFree(d_Histogram);
  cudaFree(d_Data);
}

void opt_copyFromDevice(unsigned int* h_Histogoram)
{
    cudaMemcpy(h_Histogoram, d_Histogram, HISTO_HEIGHT * HISTO_WIDTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}


/* Include below the implementation of any other functions you need */

