#include <stdio.h>
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define BLOCK_SIZE 32


__global__ void shared_kernel(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {

    const uint rowC = blockIdx.x;
    const uint colC = blockIdx.y;

    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];

    const uint trow = threadIdx.x / BLOCK_SIZE;
    const uint tcol = threadIdx.x % BLOCK_SIZE;

        A += rowC * BLOCK_SIZE * K;
        B += colC * BLOCK_SIZE;
        C += rowC * BLOCK_SIZE * N + colC * BLOCK_SIZE;

    float accum = 0.0f;

    for (int blk = 0; blk < K; blk += BLOCK_SIZE) {

        sA[trow * BLOCK_SIZE + tcol] = A[trow * K + tcol];
        sB[trow * BLOCK_SIZE + tcol] = B[trow * N + tcol];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for (int idot = 0; idot < BLOCK_SIZE; idot++) {
            accum += sA[trow * BLOCK_SIZE + idot] * sB[idot * BLOCK_SIZE + tcol];
        }

        __syncthreads();
    }

    C[trow * N + tcol] = alpha * accum + beta * C[trow * N + tcol];
}
