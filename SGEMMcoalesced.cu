
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>

__global__ void c_GEMM(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cx = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int cy = blockIdx.x * BLOCKSIZE + (threadIdx.y %  BLOCKSIZE);

    if (cx < M && cy < N) {
        float val = 0.0f;

        for (j = 0; j < K; j++) {
            val += M[cx * K + j] * N[j * N + cy];
        }
        C[cx * N + cy] = alpha * val + beta * C[cx * N + j];
    }

} 
