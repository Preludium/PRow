#pragma once
int MATRIX_SIZE = 1024;

void constantInit(float* data, int size, float val);
void checkIfCudaError(cudaError_t error, std::string function, int line);
void checkIfCudaErrorEvent(cudaError_t error, std::string function);
void computeAndPrintPerformance(float msecTotal, int nIter, dim3 dimsA, dim3 dimsB, dim3 threads);
bool checkCorrectness(dim3 dimsA, dim3 dimsC, float* h_C, float valB);
void printHelpMode(int argc, char** argv);