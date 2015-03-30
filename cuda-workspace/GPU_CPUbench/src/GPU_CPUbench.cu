#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

__global__ void vectorMultGPU(float *a, float *b, float *c, int n)
{
	int i = threadIdx.x;

	if (i < n)
	{
		c[i] = a[i] * b[i];
	}
}

void vectorMultCPU(float *a, float *b, float *c, int n)
{
	int i;

	for (i = 0; i < n; ++i)
	{
		c[i] = a[i] * b[i];
	}
}

int main()
{
	float *a, *b, *c, *GPUout;
	float *d_a, *d_b, *d_c;
	int i;

	time_t curTime, baseTime;

	a = (float*)malloc(SIZE*sizeof(float));
	b = (float*)malloc(SIZE*sizeof(float));
	c = (float*)malloc(SIZE*sizeof(float));
	GPUout = (float*)malloc(SIZE*sizeof(float));

	cudaMalloc(&d_a, SIZE*sizeof(float));
	cudaMalloc(&d_b, SIZE*sizeof(float));
	cudaMalloc(&d_c, SIZE*sizeof(float));

	for(i = 0; i < SIZE; ++i)
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE*sizeof(float), cudaMemcpyHostToDevice);

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + 10) //Runs for about 10 seconds
	{
		vectorMultCPU(a, b, c, SIZE);
		vectorMultGPU<<< 1, SIZE >>>(d_a, d_b, d_c, SIZE);
		curTime = time(NULL);
	}

	cudaMemcpy(GPUout, d_c, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < 20; ++i)
	{
		printf("CPU[%d] = %f, GPU[%d] = %f\n", i, c[i], i, GPUout[i]);
	}

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
