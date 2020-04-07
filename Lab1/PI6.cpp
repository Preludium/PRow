#include <stdio.h>
#include <omp.h>
#include <time.h>

long long num_steps = 1000000000;
double step;
#define NUM_THREADS 8

int main(int argc, char* argv[])
{
	double start, end;
	double pi = 0.0;
	volatile double sum[NUM_THREADS];
	int i;
	step = 1. / (double) num_steps;
	omp_set_num_threads(NUM_THREADS);
	start = omp_get_wtime();

#pragma omp parallel
	{
		double x;
		int id = omp_get_thread_num();
#pragma omp for
		for (i = 0, sum[id] = 0.0; i < num_steps; i++) {
			x = (i + 0.5) * step;
			sum[id] += 4.0 / (1.0 + x * x);
		}
	}

	for (int k = 0; k < NUM_THREADS; k++) {
		pi += sum[k] * step;
	}

	end = omp_get_wtime();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", (end - start));
	return 0;
}
