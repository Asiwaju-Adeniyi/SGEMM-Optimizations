#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


__global__ void mmaVectorized_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {

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

    const uint inRowA = threadIdx.x / (BK / 4);
    const uint inColA = threadIdx.x % (BK / 4);
    const uint strideA = (threadsPerBlock * 4) / BK;

    const uint inRowB = threadIdx.x / (BN /4);
    const uint inColB = threadIdx.x % (BN / 4);
    const uint strideB = (threadsPerBlock * 4) / BN;

    float tResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        float4 accum = 
        reinterpret_cast<float4 *>(&sA[inRowA * K + inColA * 4])[0];
        sA[(inColA * 4 + 0) * BM + inRowA] = accum.x;
        sA[(inColA * 4 + 1) * BM + inRowA] = accum.y;
        sA[(inColA * 4 + 2) * BM + inRowA] = accum.z;
        sA[(inColA * 4 + 2) * BM + inRowA] = accum.w;

      reinterpret_cast<float4 *> (&sB[inRowB * BN + inColB * 4])[0] = reinterpret_cast<float4 *>(&B[inRowB * N + inColB * 4])[0];

    __syncthreads();


    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i) {
            regM[i] = sA[dotIdx * BM + tRow * TM + i];
        }
    for (uint i = 0; i < TN; ++i) {
        regN[i] = sB[dotIdx * BN + tCol * TN + i];
    }  
    for (uint rIm = 0; rIm < TM ; ++rIm) {
        for (uint rIn = 0; rIn < TN; ++rIn) {
            tResults[rIm * TN + rIn] += regM[rIm] * regN[rIn];
        }
    }
    
    }
    __syncthreads();

    }

    for (uint rIm = 0; rIm < TM; rIm += 1) {
        for (uint rIn = 0; rIn < TN; rIn += 4) {
            float4 accum = reinterpret_cast<float4 *> (&C[(tRow * TM + rIn) * N + tCol * TN + rIn])[0];
            
            accum.x = alpha * tResults[rIm * TN + rIn] + beta * accum.x;
            accum.y = alpha * tResults[rIm * TN + rIn + 1] + beta * accum.y;
            accum.z = alpha * tResults[rIm * TN + rIn + 2] + beta * accum.z;
            accum.w = alpha * tResults[rIm * TN + rIn + 3] + beta * accum.w;

            reinterpret_cast<float4 *> (&C[(tRow * TM + rIn) * N + tCol * TN + rIn])[0] = accum;
        }
    }
}
