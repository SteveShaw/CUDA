texture<short, 2, cudaReadModeElementType> tex_inp;



__device__ void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

__device__ inline void compare(int* a, int* b)
{
	int temp;
	if(*a > *b)
	{
		temp = *a;
		*b = *a;
		*a = temp;
	}
}

#define min3(a,b,c) compare(a,b); compare(a,c);
#define max3(a,b,c) compare(b,c); compare(a,c);
#define minmax3(a,b,c) max3(a,b,c); compare(a,b);
#define minmax4(a,b,c,d) compare(a,b);compare(c,d);compare(a,c);compare(b,d);
#define minmax5(a,b,c,d,e) compare(a,b);compare(c,d);min3(a,c,e);max3(b,d,e);
#define minmax6(a,b,c,d,e,f) compare(a,d);compare(b,e);compare(c,f);min3(a,b,c);max3(b,e,f);


__global__ void kernel_median_w3(short* output, int width, int height)
{
	int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int i = __mul24(blockIdx.y,blockDim.y)+threadIdx.y;
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
	minmax6(&r0,&r1,&r2,&r3,&r4,&r5);
	r5 = tex2D(tex_inp,j-1,i+1);
	minmax5(&r1,&r2,&r3,&r4,&r5);
	r5 = tex2D(tex_inp,j,i+1);
	minmax4(&r2,&r3,&r4,&r5);
	r5 = tex2D(tex_inp,j+1,i+1);
	minmax3(&r3,&r4,&r5);
	output[__mul24(i,width)+j] = r4;
}
