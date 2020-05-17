// System includes
#define WIN32
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#pragma once

//	constants
#define CONST_MATRIX_SIZE 32;
#define CONST_SIZE_OF_BLOCK 32;
#define MAX_ERROR 1e-5;

void constantInit(float* data, int size, float val);
void checkIfCudaError(cudaError_t error, std::string function, int line);
void checkIfCudaErrorEvent(cudaError_t error, std::string function);
void computeAndPrintPerformance(float msecTotal, int nIter, dim3 dimsA, dim3 dimsB, dim3 threads);
bool checkCorrectness(dim3 dimsA, dim3 dimsC, float* h_C, float valB);
void printHelpMode(int argc, char** argv);
void setDimIfCmdArgument(int argc, char** argv, const char* argument, unsigned int &variable);
void setDeviceIfCmdArgument(int argc, char** argv, const char* argument, int& variable);
void checkComputeMode(cudaDeviceProp deviceProp);
void checkDimensions(dim3 dimsA, dim3 dimsB);