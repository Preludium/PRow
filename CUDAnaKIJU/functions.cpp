#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <string>

void constantInit(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

void checkIfCudaError(cudaError_t error, std::string function, int line) {
    if (error != cudaSuccess)
    {
        std::cout << function << " returned error code " << error << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkIfCudaErrorEvent(cudaError_t error, std::string event) {
    if (error != cudaSuccess)
    {
        std::cout << "Failed to " << event << ". Error code " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void computeAndPrintPerformance(float msecTotal, int nIter, dim3 dimsA, dim3 dimsB, dim3 threads) {
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y
    );
}

bool checkCorrectness(dim3 dimsA, dim3 dimsC, float* h_C, float valB) {
    bool result = true;
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        bool isErrorTooHigh = fabs(h_C[i] - (dimsA.x * valB)) > 1e-5;
        if (isErrorTooHigh)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C[i], dimsA.x * valB);
            result = false;
        }
    }
    return result;
}

void printHelpMode(int argc, char **argv) {
    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }
}