#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>

__global__ void vectorMult(float *a, float *b, float *c, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while (i < n)
	{
		c[i] = a[i] * b[i];
		i+= blockDim.x * gridDim.x;
	}
}

int main(int argc, char **argv)
{
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int i;

	int size = 100000;
	int elapsed_time = 10;
	int option;

	while ((option = getopt (argc, argv, "s:t:")) != -1)
	{
		switch (option)
		{
		case 's':
			size = atoi(optarg);
			break;
		case 't':
			elapsed_time = atoi(optarg);
			break;
		case '?':
			if (optopt == 's' || optopt == 't')
				fprintf (stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint (optopt))
				fprintf (stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf (stderr,
					   "Unknown option character `\\x%x'.\n",
					   optopt);
			return 1;
		default:
			abort ();
		}
	}

	time_t curTime, baseTime;

	a = (float*)malloc(size*sizeof(float));
	b = (float*)malloc(size*sizeof(float));
	c = (float*)malloc(size*sizeof(float));

	cudaMalloc(&d_a, size*sizeof(float));
	cudaMalloc(&d_b, size*sizeof(float));
	cudaMalloc(&d_c, size*sizeof(float));

	for(i = 0; i < size; i++)
	{
		a[i] = b[i] = (float)i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size*sizeof(float), cudaMemcpyHostToDevice);

	int count = 0;

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + elapsed_time)
	{
		count++;
		cudaDeviceSynchronize();
		vectorMult<<< (size+511)/512, 512 >>>(d_a, d_b, d_c, size);
		curTime = time(NULL);
	}

	cudaMemcpy(c, d_c, size*sizeof(float), cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("Test Complete");

	return 0;
}

