#include <stdio.h>
#include <omp.h>
#include <time.h>

long long num_steps = 1000000000;
double step;
#define NUM_THREADS 2

int main(int argc, char* argv[])
{
	double start, end;
	double x, pi;
	volatile double sum[51];
	int i;
	step = 1. / (double)num_steps;
	omp_set_num_threads(NUM_THREADS);
	
	for(int j = 0; j < 50; j++) {
		start = omp_get_wtime();

#pragma omp parallel
		{
			int id = omp_get_thread_num();
#pragma omp for
			for (i = 0, sum[j + id] = 0.0; i < num_steps; i++) {
				x = (i + 0.5) * step;
				sum[j + id] += 4.0 / (1.0 + x * x);
			}
		}
		
		pi = 0;
		for (int k = j; k < j + NUM_THREADS; k++) {
			pi += sum[k] * step;
		}

		end = omp_get_wtime();

		printf("Iteracja %d\n", j + 1);
		printf("Wartosc liczby PI wynosi %15.12f\n", pi);
		printf("Czas przetwarzania wynosi %f sekund\n\n", (end - start));
	}
	return 0;
}