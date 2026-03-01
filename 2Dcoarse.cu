#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))



__global__ void tCoarse2D_kernel(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint resultsPerBlock = BM * BN;
    const uint threadsPerBlock = resultsPerBlock / TM * TN;

    assert(threadsPerBlock == blockDim.x);

    const uint tRow = threadIdx.x / (BN / TN);
    const uint tCol = threadIdx.x % (BN / TN);

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint inRowA = threadIdx.x / BK;
    const uint inColA = threadIdx.x % BK;
    const uint strideA = threadsPerBlock / BK;

    const uint inRowB = threadIdx.x / BN;
    const uint inColB = threadIdx.x % BN;
    const uint strideB = threadsPerBlock / BN;

    float tResAccum[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int offset = 0; offset < BM; offset += strideA) {
            sA[(inRowA + offset) * BK + inColA] = A[(inRowA + offset) * K + inColA];
        }

        for (int offset = 0; offset < BK; offset += strideB) {
             sB[(inRowB + offset) * BN + inColB] = B[(inRowB + offset) * N + inColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dI = 0; dI < BK; dI++) {
             for (uint i = 0; i < TM; ++i) {
                regM[i] = sA[(tRow * TM + i) * BK + dI];
             }

             for (uint i = 0; i < TN; ++i) {
                regN[i] = sB[dI * TN + tCol * TN + i];
             }

             for (uint rIm = 0; rIm < TM; rIm++) {
                for (uint rIn = 0; rIn < TN; rIn++) {
                    tResAccum[rIm * TN + rIn] += regM[rIm] * regN[rIn];
                }
             }
        }
       __syncthreads();
    }  

    for (uint rIm = 0; rIm < TM; rIm++) {
        for (uint rIn = 0; rIn < TN; rIn++) {
            C[(tRow * TM + rIn) * N + tCol * TN + rIn] = alpha * tResAccum[rIm * TN + rIn] + beta * C[(tRow * TM + rIn) * N + tCol * TN + rIn];
        }
    }
}
