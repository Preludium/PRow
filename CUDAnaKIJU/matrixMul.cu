/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */
#include <Constants.h>

#define CONST_MATRIX_SIZE 1024;
#define CONST_SIZE_OF_BLOCK 8;
#define N_ITER 300

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Pobranie pierwszego bloku danych z pamiêci do A
    // tu bêdziemy zmieniaæ __shared__ na rejestry
    float AAs;
    float ABs;

    AAs = A[aBegin + wA * ty + tx];
    ABs = B[bBegin + wB * ty + tx];
    /*
    * Loop over all the sub-matrices of A and B
    * required to compute the block sub-matrix
    */

    // (CHYBA!) trzeba zacz¹æ od kolejnego bloku, bo A wczeœniej ju¿ pobra³o sobie pierwszy??
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        /*
        * Declaration of the shared memory array As used to
        * store the sub-matrix of A
        */
        __shared__ float BAs[BLOCK_SIZE][BLOCK_SIZE];
        /*
        * Declaration of the shared memory array Bs used to
        * store the sub-matrix of B
        */
        __shared__ float BBs[BLOCK_SIZE][BLOCK_SIZE];

        // Przepisanie z A na wspó³dzielon¹ B
        // CHYBA(!) to bêdziemy zamieniaæ z obliczeniami ale nwm
        // -pytanie: czy w jakiœ spoœób musimy "zwalniaæ" A? (tak jest na wyk³adzie)
        BAs[ty][tx] = AAs;
        BBs[ty][tx] = ABs;

        //printf("ABS: %f BBS: %f\n", ABs[ty][tx], BBs[ty][tx]);
        
        float AAs;
        float ABs;

        /*
        * Load the matrices from device memory
        * to shared memory; each thread loads
        * one element of each matrix
        */

        /*
        * Synchronize to make sure the matrices are loaded
        */
        __syncthreads();

        // Pobranie kolejnego bloku danych z pamiêci globalnej do A
        if (a != aEnd) {
            AAs = A[a + aStep + wA * ty + tx];
            ABs = B[b + bStep + wB * ty + tx];
        }
        /*
        * Multiply the two matrices together;
        * each thread computes one element
        * of the block sub-matrix
        */
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            // Obliczenia, ale tym razem na B
            Csub += BAs[ty][k] * BBs[k][tx];
        }

        /*
        * Synchronize to make sure that the preceding
        * computation is done before loading two new
        * sub-matrices of A and B in the next iteration
        */

        //pytanie: to samo co wczeœniej, na wyk³¹dzie jest zwolnienie B, trzeba to robiæ czy siê samo robi?
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);
    

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);
    checkIfCudaError(error, "cudaMalloc d_A", __LINE__);

    error = cudaMalloc((void **) &d_B, mem_size_B);
    checkIfCudaError(error, "cudaMalloc d_B", __LINE__);

    error = cudaMalloc((void **) &d_C, mem_size_C);
    checkIfCudaError(error, "cudaMalloc d_C", __LINE__);
   
    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    checkIfCudaError(error, "cudaMemcpy (d_A, h_A)", __LINE__);

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    checkIfCudaError(error, "cudaMemcpy (d_B, h_B)", __LINE__);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16)
    {
        matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);
    checkIfCudaErrorEvent(error, "cudaEventCreate(start)");

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);
    checkIfCudaErrorEvent(error, "cudaEventCreate(stop)");


    // Record the start event
    error = cudaEventRecord(start, NULL);
    checkIfCudaErrorEvent(error, "cudaEventRecord(start)");

    // Execute the kernel
    int nIter = N_ITER;



    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
            matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);
    checkIfCudaErrorEvent(error, "cudaEventRecord(stop)");

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);
    checkIfCudaErrorEvent(error, "cudaEventSynchronize(stop)");

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    checkIfCudaErrorEvent(error, "cudaEventGetElapsedTime(start, stop)");

    // Compute and print the performance
    computeAndPrintPerformance(msecTotal, nIter, dimsA, dimsB, threads);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    checkIfCudaError(error, "cudaMemcpy(h_C, d_C)", __LINE__);

    printf("Checking computed result for correctness: ");
    bool correct = checkCorrectness(dimsA, dimsC, h_C, valB);

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

    cudaDeviceReset();
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    printHelpMode(argc, argv);

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;
    setDeviceIfCmdArgument(argc, argv, "device", devID);

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    checkComputeMode(deviceProp);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = CONST_SIZE_OF_BLOCK;
    int matrix_size = CONST_MATRIX_SIZE;
    dim3 dimsA(matrix_size, matrix_size, 1);
    dim3 dimsB(matrix_size, matrix_size, 1);

    // width of Matrix A
    setDimIfCmdArgument(argc, argv, "wA", dimsA.x);
    // height of Matrix A
    setDimIfCmdArgument(argc, argv, "hA", dimsA.y);
    // width of Matrix B
    setDimIfCmdArgument(argc, argv, "wB", dimsB.x);
    // height of Matrix B
    setDimIfCmdArgument(argc, argv, "hB", dimsB.y);

    checkDimensions(dimsA, dimsB);

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    printf("Block size: %d\n", block_size);

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}
