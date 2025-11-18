
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void naiveGeMM(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < M && y < N) {
        float tmp = 0.0; 

        for (int i = 0; i < K; i++) {
            tmp += M[x * K + i] * N[N * i + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
} 
