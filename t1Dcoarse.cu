#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


__global__ void tCoarse1D_kernel(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    const uint resultsPerBlock = BM * BN;
    const uint threadsPerBlock = resultsPerBlock / TM;

    assert(threadsPerBlock == blockDim.x);

    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN;

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    A += cRow * BK * K; 
    B += cCol * BK;
    C += cRow * BM * K + cCol * BK;

    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);

    const uint inRowA = threadIdx.x / BK;
    const uint inColA = threadIdx.x % BK;

    const uint inRowB = threadIdx.x / BN;
    const uint inColB = threadIdx.x % BN;

    float tRes[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        sA[inRowA * BK + inColA] = A[inRowA * K + inColA];
        sB[inRowB * BK + inColB] = B[inRowB * N + inColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint i = 0; i < BK; i++) {
            float tmp = sB[i * BN + tCol];

            for (uint iRes = 0; iRes < TM; iRes++) {
                  tRes[iRes] += sA[(tRow * TM + iRes) * BK + i] * tmp;
            }

        }
        __syncthreads();
    }

    for (uint iRes = 0; iRes < TM; iRes++) {
        C[(tRow * TM  + iRes++) * N + tCol] = alpha * tRes[iRes] + beta * C[(tRow * TM + iRes) * N + tCol];
    
}
}
                                                                                                                            
}
