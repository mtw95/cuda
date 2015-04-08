#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <ctype.h>

struct ThreadStruct {
	float *a, *b, *c;
	int size, time;
};

void vectorMult(float *a, float *b, float *c, int n)
{
	int i;

	for (i = 0; i < n; ++i)
	{
		c[i] = a[i] * b[i];
	}
}

void threads(struct ThreadStruct *data)
{
	time_t curTime, baseTime;

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + data->time) //Runs for 10 seconds
	{
		vectorMult(data->a, data->b, data->c, data->size);
		curTime = time(NULL);
	}
}



int main(int argc, char **argv)
{
	int cores = 4;
	int size = 100000;
	int time = 10;
	int option;

	while ((option = getopt (argc, argv, "s:t:c:")) != -1)
	{
		switch (option)
		{
		case 's':
			size = atoi(optarg);
			break;
		case 't':
			time = atoi(optarg);
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

	pthread_t *thread = (pthread_t*)malloc(cores*sizeof(pthread_t));

	float *a, *b, *c;
	int i;


	a = (float*)malloc(size*sizeof(float));
	b = (float*)malloc(size*sizeof(float));
	c = (float*)malloc(size*sizeof(float));

	for (i = 0; i < size; ++i)
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	struct ThreadStruct data = {a, b, c, size, time};

	for (i = 0; i < cores; ++i)
		pthread_create(&thread[i], NULL, threads, &data);

	for (i = 0; i < cores; ++i)
		pthread_join(thread[i],NULL);


	free(a);
	free(b);
	free(c);

	return 0;
}
