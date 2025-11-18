
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void naiveGeMM(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int y = blockIdx.x * BLOCKSIZE + (threadIdx.y %  BLOCKSIZE);

    if (x < M && y < N) {
        float val = 0.0f;

        for (j = 0; j < K; j++) {
            val += M[x * K + j] + N[j * N + y];
        }
        C[x * K + j] = alpha * val + beta[row * K + j];
    }

} 
