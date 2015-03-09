texture<short, 2, cudaReadModeElementType> tex_inp;

__device__ void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}


__global__ void kernel_median_w3(short* output, int width, int height)
{
	int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int i = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	int r0,r1,r2,r3,r4,r5,r6,r7,r8;
	r0 = tex2D(tex_inp,j-1,i-1);
	r1 = tex2D(tex_inp,j,i-1);
	r2 = tex2D(tex_inp,j+1,i-1);
	r3 = tex2D(tex_inp,j-1,i);
	r4 = tex2D(tex_inp,j,i);
	r5 = tex2D(tex_inp,j+1,i);
	r6 = tex2D(tex_inp,j-1,i+1);
	r7 = tex2D(tex_inp,j,i+1);
	r8 = tex2D(tex_inp,j+1,i+1);

	bool stop = false;
	
	while(!stop)
	{
		stop = true;
		if(r0<r1)
		{
			stop = false;
			swap(&r0,&r1);
		}
		
		if(r1<r2)
		{
			stop = false;
			swap(&r1,&r2);
		}

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
		
			
		if(r5<r6)
		{
			stop = false;
			swap(&r5,&r6);
		}
		
		
		if(r6<r7)
		{
			stop = false;
			swap(&r6,&r7);
		}
		
		if(r7<r8)
		{
			stop = false;
			swap(&r7,&r8);
		}
	}
	output[__mul24(i,width)+j] = r4;
}

__global__ void kernel_filter(short* output, int width, int height, int r)
{
	 int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	 int i = __mul24(blockIdx.y,blockDim.y)+threadIdx.y;
	// output[__mul24(i,width)+j] = tex2D(tex_inp,j,i);
	
	if(j<width && i<height)
	{
		short bin, idx_i, idx_j;
		short histogram[256];
		
		for(bin=0;bin<256;bin++) 
		{
			histogram[bin] = 0;
		}
		
		for(idx_i=i-r;idx_i<=i+r;++idx_i)
		{
			for(idx_j=j-r;idx_j<=j+r;++idx_j)
			{
				histogram[tex2D(tex_inp,idx_j,idx_i)]++;
			}
		}
		
		short center_pixel = 0;
		
		for(bin=0;bin<256;++bin)
		{
			center_pixel += histogram[bin];
			if(center_pixel>((2*r+1)*(2*r+1))>>1) break;
		}
		
		output[__mul24(i,width)+j] = bin;
	}
	
}

__device__ inline void CompareAndSwap(int* a, int* b)
{
	int temp;
	if(*a > *b)
	{
		temp = *a;
		*b = *a;
		*a = temp;
	}
}


__global__ void kernel_median_w3minr(short* output, int width, int height)
{
	int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int i = __mul24(blockIdx.y,blockDim.y)+threadIdx.y;
	int r0,r1,r2,r3,r4,r5;
	
	r0 = tex2D(tex_inp,j-1,i-1);
	r1 = tex2D(tex_inp,j,i-1);
	r2 = tex2D(tex_inp,j+1,i-1);
	r3 = tex2D(tex_inp,j-1,i);
	r4 = tex2D(tex_inp,j,i);
	r5 = tex2D(tex_inp,j+1,i);
	
	bool stop = false;
	while(!stop)
	{
		stop = true;
		if(r0<r1)
		{
			stop = false;
			swap(&r0,&r1);
		}
		
		if(r1<r2)
		{
			stop = false;
			swap(&r1,&r2);
		}

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}

	
	// CompareAndSwap(&r0,&r3);
	// CompareAndSwap(&r1,&r4);
	// CompareAndSwap(&r2,&r5);
	// CompareAndSwap(&r0,&r1);
	// CompareAndSwap(&r0,&r2);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r1,&r5);
	
	r5 = tex2D(tex_inp,j-1,i+1);
	
	stop = false;
	while(!stop)
	{
		stop = true;
	
		if(r1<r2)
		{
			stop = false;
			swap(&r1,&r2);
		}

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}

	// CompareAndSwap(&r1,&r2);
	// CompareAndSwap(&r3,&r4);
	// CompareAndSwap(&r1,&r3);
	// CompareAndSwap(&r1,&r5);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r2,&r5);
	
	r5 = tex2D(tex_inp,j,i+1);
	stop = false;
	while(!stop)
	{
		stop = true;

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}
	
	// CompareAndSwap(&r2,&r3);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r2,&r4);
	// CompareAndSwap(&r3,&r5);
	
	r5 = tex2D(tex_inp,j+1,i+1);
	
	stop = false;
	while(!stop)
	{
		stop = true;
	
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}	

	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r3,&r5);
	// CompareAndSwap(&r3,&r4);

	output[__mul24(i,width)+j] = r4;
}

__global__ void kernel_median_2pass(short* output, int width, int height)
{
	int j = __mul24(__mul24(blockIdx.x,blockDim.x) + threadIdx.x,2);
	int i = __mul24(blockIdx.y,blockDim.y)+threadIdx.y;
	int r0,r1,r2,r3,r4,r5;
	int g0,g1,g2,g3,g4,g5;
	
	r0 = tex2D(tex_inp,j,i-1);
	r1 = tex2D(tex_inp,j+1,i-1);
	r2 = tex2D(tex_inp,j,i);
	r3 = tex2D(tex_inp,j+1,i);
	r4 = tex2D(tex_inp,j,i+1);
	r5 = tex2D(tex_inp,j+1,i+1);
	
	bool stop = false;
	while(!stop)
	{
		stop = true;
		if(r0<r1)
		{
			stop = false;
			swap(&r0,&r1);
		}
		
		if(r1<r2)
		{
			stop = false;
			swap(&r1,&r2);
		}

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}
	
	// CompareAndSwap(&r0,&r3);
	// CompareAndSwap(&r1,&r4);
	// CompareAndSwap(&r2,&r5);
	// CompareAndSwap(&r0,&r1);
	// CompareAndSwap(&r0,&r2);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r1,&r5);
	
	g0=r0;g1=r1;g2=r2;
	g3=r3;g4=r4;g5=r5;

	r5 = tex2D(tex_inp,j-1,i);
	g5 = tex2D(tex_inp,j+2,i);
	
	stop = false;
	while(!stop)
	{
		stop = true;
		
		if(r1<r2)
		{
			stop = false;
			swap(&r1,&r2);
		}

		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}
	
	stop = false;
	while(!stop)
	{
		stop = true;
	
		if(g1<g2)
		{
			stop = false;
			swap(&g1,&g2);
		}

		if(g2<g3)
		{
			stop = false;
			swap(&g2,&g3);
		}
		
		if(g3<g4)
		{
			stop = false;
			swap(&g3,&g4);
		}
		
		if(g4<g5)
		{
			stop = false;
			swap(&g4,&g5);
		}
	}
	
	// CompareAndSwap(&r1,&r2);
	// CompareAndSwap(&r3,&r4);
	// CompareAndSwap(&r1,&r3);
	// CompareAndSwap(&r1,&r5);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r2,&r5);

	// CompareAndSwap(&g1,&g2);
	// CompareAndSwap(&g3,&g4);
	// CompareAndSwap(&g1,&g3);
	// CompareAndSwap(&g1,&g5);
	// CompareAndSwap(&g4,&g5);
	// CompareAndSwap(&g2,&g5);

	r5 = tex2D(tex_inp,j-1,i-1);
	g5 = tex2D(tex_inp,j+2,i-1);
	
	stop = false;
	while(!stop)
	{
		stop = true;
		
		if(r2<r3)
		{
			stop = false;
			swap(&r2,&r3);
		}
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}
	
	stop = false;
	while(!stop)
	{
		stop = true;
	
		if(g2<g3)
		{
			stop = false;
			swap(&g2,&g3);
		}
		
		if(g3<g4)
		{
			stop = false;
			swap(&g3,&g4);
		}
		
		if(g4<g5)
		{
			stop = false;
			swap(&g4,&g5);
		}
	}
	
	// CompareAndSwap(&r2,&r3);
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r2,&r4);
	// CompareAndSwap(&r3,&r5);
	
	// CompareAndSwap(&g2,&g3);
	// CompareAndSwap(&g4,&g5);
	// CompareAndSwap(&g2,&g4);
	// CompareAndSwap(&g3,&g5);
	
	r5 = tex2D(tex_inp,j-1,i+1);
	g5 = tex2D(tex_inp,j+2,i+1);
	
	stop = false;
	while(!stop)
	{
		stop = true;
		
		if(r3<r4)
		{
			stop = false;
			swap(&r3,&r4);
		}
		
		if(r4<r5)
		{
			stop = false;
			swap(&r4,&r5);
		}
	}
	
	stop = false;
	while(!stop)
	{
		stop = true;
		
		if(g3<g4)
		{
			stop = false;
			swap(&g3,&g4);
		}
		
		if(g4<g5)
		{
			stop = false;
			swap(&g4,&g5);
		}
	}
	
	// CompareAndSwap(&r4,&r5);
	// CompareAndSwap(&r3,&r5);
	// CompareAndSwap(&r3,&r4);
	
	// CompareAndSwap(&g4,&g5);
	// CompareAndSwap(&g3,&g5);
	// CompareAndSwap(&g3,&g4);
	
	output[__mul24(i,width)+j] = r4;
	output[__mul24(i,width)+j+1] = g4;
}
