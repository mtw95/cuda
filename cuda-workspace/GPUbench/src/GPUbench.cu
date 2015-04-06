#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 100000

__global__ void vectorMult(float *a, float *b, float *c, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while (i < n)
	{
		c[i] = a[i] * b[i];
		i+= blockDim.x * gridDim.x;
	}
}

int main()
{
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int i;

	time_t curTime, baseTime;

	a = (float*)malloc(SIZE*sizeof(float));
	b = (float*)malloc(SIZE*sizeof(float));
	c = (float*)malloc(SIZE*sizeof(float));

	cudaMalloc(&d_a, SIZE*sizeof(float));
	cudaMalloc(&d_b, SIZE*sizeof(float));
	cudaMalloc(&d_c, SIZE*sizeof(float));

	for(i = 0; i < SIZE; i++)
	{
		a[i] = b[i] = (float)i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE*sizeof(float), cudaMemcpyHostToDevice);

	int count = 0;

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + 10) //Runs for about 10 seconds
	{
		count++;
		cudaDeviceSynchronize();
		vectorMult<<< (SIZE+511)/512, 512 >>>(d_a, d_b, d_c, SIZE);
		curTime = time(NULL);
	}

	cudaMemcpy(c, d_c, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	printf("Call Count: %d\n", count);
	for (i = 0; i < 10; ++i)
	{
		printf("c[%d] = %f\n", i, c[i]);
	}
	printf("c[99,999] = %f\n", c[SIZE-1]);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

