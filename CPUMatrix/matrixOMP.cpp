// Implementacja mno�enia macierzy jest realizowana za pomoca typowego 
// algorytmu podr�cznikowego. 
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <omp.h>

#define USE_MULTIPLE_THREADS true
#define MAXTHREADS 128
int NumThreads;
double start;

static const int ROWS = 1000;     // liczba wierszy macierzy
static const int COLUMNS = 1000;  // lizba kolumn macierzy

float matrix_a[ROWS][COLUMNS];    // lewy operand 
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik


void initialize_matrices() {
    // zdefiniowanie zawarosci poczatkowej macierzy
//#pragma omp parallel for 
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix_a[i][j] = (float)rand() / RAND_MAX;
            matrix_b[i][j] = (float)rand() / RAND_MAX;
            matrix_r[i][j] = 0.0;
        }
    }
}

void initialize_matricesZ() {
    // zdefiniowanie zawarosci poczatkowej macierzy
#pragma omp parallel for 
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            matrix_r[i][j] = 0.0;
        }
    }
}

void multiply_matrices_KIJ_sequential()
{
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_IJK() {
#pragma omp parallel for 
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            float sum = 0.0;
            for (int k = 0; k < COLUMNS; k++) {
                sum = sum + matrix_a[i][k] * matrix_b[k][j];
            }
            matrix_r[i][j] = sum;
        }
    }
}

void multiply_matrices_IKJ() {
#pragma omp parallel for 
    for (int i = 0; i < ROWS; i++)
        for (int k = 0; k < COLUMNS; k++)
            for (int j = 0; j < COLUMNS; j++)
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_JIK() {
#pragma omp parallel for 
    for (int j = 0; j < COLUMNS; j++) {
        for (int i = 0; i < ROWS; i++) {
            float sum = 0.0;
            for (int k = 0; k < COLUMNS; k++) {
                sum = sum + matrix_a[i][k] * matrix_b[k][j];
            }
            matrix_r[i][j] = sum;
        }
    }
}

void multiply_matrices_JKI() {
#pragma omp parallel for 
    for (int j = 0; j < COLUMNS; j++)
        for (int k = 0; k < COLUMNS; k++)
            for (int i = 0; i < ROWS; i++)
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_parallel_k() {
#pragma omp parallel 
#pragma omp for
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_parallel_i() {
#pragma omp parallel 
	for (int k = 0; k < COLUMNS; k++)
#pragma omp for
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_parallel_j() {
#pragma omp parallel 
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
#pragma omp for
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_atomic()
{
#pragma omp parallel for
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
#pragma omp atomic
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}


void print_elapsed_time() {
    double elapsed;
    double resolution;

    // wyznaczenie i zapisanie czasu przetwarzania
    elapsed = (double)clock() / CLK_TCK;
    resolution = 1.0 / CLK_TCK;
    printf("Czas: %8.4f sec \n", elapsed - start);

}

int main(int argc, char* argv[]) {
    //	 start = (double) clock() / CLK_TCK ;
    
    //Determine the number of threads to use
    if (USE_MULTIPLE_THREADS) {
        SYSTEM_INFO SysInfo;
        GetSystemInfo(&SysInfo);
        NumThreads = SysInfo.dwNumberOfProcessors;
        if (NumThreads > MAXTHREADS)
            NumThreads = MAXTHREADS;
    }
    else
        NumThreads = 1;
    printf("liczba watkow  = %d\n\n", NumThreads);

    initialize_matrices();

    start = (double)clock() / CLK_TCK;
    multiply_matrices_KIJ_sequential();
    printf("KIJ_sequential ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_IJK();
    printf("IJK ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_IKJ();
    printf("IKJ ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_JIK();
    printf("JIK ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_JKI();
    printf("JKI ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_KIJ_parallel_k();
    printf("KIJ_parallel_k ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_KIJ_parallel_i();
    printf("KIJ_parallel_i ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_KIJ_parallel_j();
    printf("KIJ_parallel_j ");
    print_elapsed_time();

    initialize_matricesZ();
    start = (double)clock() / CLK_TCK;
    multiply_matrices_KIJ_atomic();
    printf("KIJ_atomic ");
    print_elapsed_time();

    return(0);
}