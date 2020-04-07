#include <stdio.h>
#include <omp.h>
#include <time.h>

long long num_steps = 1000000000;
double step;

int main(int argc, char* argv[])
{
	double start, end;
	double x, pi, sum = 0.0;
	int i;
	step = 1. / (double)num_steps;
	start = omp_get_wtime();
	omp_set_num_threads(2);

#pragma omp parallel for
	for (i = 0; i < num_steps; i++)
	{
		x = (i + .5) * step;
		sum += 4.0 / (1. + x * x);
	}

	pi = sum * step;
	end = omp_get_wtime();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", (end - start));
	return 0;
	
}