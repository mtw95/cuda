#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

void vectorMult(float *a, float *b, float *c, int n)
{
	int i;

	for (i = 0; i < n; ++i)
	{
		c[i] = a[i] * b[i];
	}
}

int main()
{
	float *a, *b, *c;
	int i;

	time_t curTime, baseTime;

	a = (float*)malloc(SIZE*sizeof(float));
	b = (float*)malloc(SIZE*sizeof(float));
	c = (float*)malloc(SIZE*sizeof(float));

	for (i = 0; i < SIZE; ++i)
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	baseTime = curTime = time(NULL);
	while(curTime < baseTime + 10) //Runs for about 10 seconds
	{
		vectorMult(a, b, c, SIZE);
		curTime = time(NULL);
	}

	for (i = 0; i < 20; ++i)
	{
		printf("c[%d] = %f\n", i, c[i]);
	}

	free(a);
	free(b);
	free(c);

	return 0;
}
