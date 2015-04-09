#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <ctype.h>

struct ThreadStruct {
	float *a, *b, *c;
	int size, elapsed_time;
};

__global__ void vectorMultGPU(float *a, float *b, float *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	while (i < n)
	{
		c[i] = a[i] * b[i];
		i+= blockDim.x * gridDim.x;
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

void *threadCPU(void *threadarg)
{
	time_t curTime, baseTime;

	struct ThreadStruct *data;
	data = (struct ThreadStruct*) threadarg;

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + data->elapsed_time) //Runs for 10 seconds
	{
		vectorMultCPU(data->a, data->b, data->c, data->size);
		curTime = time(NULL);
	}
	return NULL;
}

int main(int argc, char **argv)
{
	int cores = 4;
	int size = 100000;
	int elapsed_time = 10;
	int option;

	while ((option = getopt (argc, argv, "s:t:c:")) != -1)
	{
		switch (option)
		{
		case 's':
			size = atoi(optarg);
			break;
		case 't':
			elapsed_time = atoi(optarg);
			break;
		case 'c':
			cores = atoi(optarg);
			break;
		case '?':
			if (optopt == 's' || optopt == 't' || optopt == 'c')
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

	pthread_t *thread_arr = (pthread_t*)malloc(cores*sizeof(pthread_t));

	float *a, *b, *c, *GPUout;
	float *d_a, *d_b, *d_c;
	int i;

	a = (float*)malloc(size*sizeof(float));
	b = (float*)malloc(size*sizeof(float));
	c = (float*)malloc(size*sizeof(float));
	GPUout = (float*)malloc(size*sizeof(float));

	cudaMalloc(&d_a, size*sizeof(float));
	cudaMalloc(&d_b, size*sizeof(float));
	cudaMalloc(&d_c, size*sizeof(float));

	for(i = 0; i < size; ++i)
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size*sizeof(float), cudaMemcpyHostToDevice);

	time_t curTime, baseTime;
	struct ThreadStruct Threaddata = {a, b, c, size, elapsed_time};

	for (i = 0; i < cores; ++i)
		pthread_create(&thread_arr[i], NULL, threadCPU, (void *) &Threaddata);

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + elapsed_time)
	{
		cudaDeviceSynchronize();
		vectorMultGPU<<< (size+511)/512, 512 >>>(d_a, d_b, d_c, size);
		curTime = time(NULL);
	}

	for (i = 0; i < cores; ++i)
		pthread_join(thread_arr[i],NULL);

	cudaMemcpy(GPUout, d_c, size*sizeof(float), cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("Test Complete\n");

	return 0;
}
