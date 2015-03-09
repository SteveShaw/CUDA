#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <iostream>
#include "kernel_filter.cu"
#include "util.h"

void filter(dim3 block, dim3 grid, short*d_out, unsigned int W, unsigned int H, int r)
{
	kernel_filter<<<grid,block,0>>>(d_out,W,H,r);
}
int main(int argc, char** argv)
{
	int option = 0;
	
	if(argc>1)
	{
		option = atoi(argv[1]);
		std::cout<<"Option="<<option<<std::endl;
	}
	std::string filename = "noise.pgm";
	short *h_in, *h_out, *d_out;

	int size, bsx=16, bsy = 16;
	dim3 dimBlock, dimGrid;

	cudaChannelFormatDesc channelD = cudaCreateChannelDesc<short>();

	//unsigned int *h_img = NULL;
	//unsigned int *h_out_img, H,W;
	unsigned char *h_img = NULL;
	unsigned char *h_out_img;
	unsigned int H, W;
	//cutilCheckError(cutLoadPGMi(filename.c_str(),&h_img,&W,&H));
	cutilCheckError(cutLoadPGMub(filename.c_str(),&h_img,&W,&H));
	std::cout<<"Input Image Width="<<W<<std::endl;
	std::cout<<"Input Image Height="<<H<<std::endl;
	size = H*W*sizeof(short);
	h_in = new short[H*W];
	for(int k = 0;k<H*W;++k)
	{
		h_in[k] = (short)h_img[k];
	}

	cudaHostAlloc((void**)&h_out,size,cudaHostAllocDefault);
	cudaMalloc((void**)&d_out,size);
	cudaArray* img_arr;
	cudaMallocArray(&img_arr,&channelD,W,H);
	cudaBindTextureToArray(tex_inp,img_arr,channelD);
	cudaMemcpyToArray(img_arr,0,0,h_in,size,cudaMemcpyHostToDevice);
	dimBlock = dim3(bsx,bsy,1);
	dimGrid = dim3(W/dimBlock.x,H/dimBlock.y,1);

	//TIME_IT("kernel_filter",1000,  filter(dimBlock,dimGrid,d_out,W,H,5);)
	unsigned int timer;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutResetTimer(timer));
	cutilCheckError(cutStartTimer(timer));
	if(option==0)
	{
		for(int count = 0;count<1000;++count)
		{
			// kernel_filter<<<dimGrid,dimBlock,0>>>(d_out,W,H,3);
			kernel_filter<<<dimGrid,dimBlock>>>(d_out,W,H,3);
		}
	}
	
	if(option==1)
	{
		for(int count = 0;count<1000;++count)
		{
			// kernel_filter<<<dimGrid,dimBlock,0>>>(d_out,W,H,3);
			kernel_median_w3<<<dimGrid,dimBlock>>>(d_out,W,H);
		}
	}
	
	if(option==2)
	{
		for(int count = 0;count<1000;++count)
		{
			// kernel_filter<<<dimGrid,dimBlock,0>>>(d_out,W,H,3);
			kernel_median_w3minr<<<dimGrid,dimBlock>>>(d_out,W,H);
		}
	}

	cudaThreadSynchronize();
	cutilCheckError(cutStopTimer(timer));
	printf ("It takes about : %f ms to run,\n" , cutGetTimerValue ( timer ) /1000);
	
	
	

	cudaMemcpy(h_out,d_out,size,cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	h_out_img = new unsigned char[H*W];
	for(int k = 0;k<H*W;++k)
	{
		h_out_img[k] = (unsigned char)h_out[k];
	}

	cutilCheckError(cutSavePGMub("test_out.pgm",h_out_img,W,H));

	cudaFreeHost(h_out);
	cudaFree(d_out);

	delete []h_out_img;
	delete []h_in;
}
