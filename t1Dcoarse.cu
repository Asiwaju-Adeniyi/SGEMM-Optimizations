#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))


__global__ void tCoarsen_kernel(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {
    
const int BM = 64;
const int BN = 64;
const int BK = 8;                                                                                                                                                                                                                                                                                                                                                                       
const int TM = 8;

     const uint cRow = threadIdx.y;
     const uint cCol = threadIdx.x;

     const uint totalBTresults = BM * BN;
     const uint numThreadsPerBlock = totalBTresults / TM;

     assert(numThreadsPerBlock == blockDim.x);

     const int tRow = threadIdx.x / BK;
     const int tCol = threadIdx.x % BK;

     __shared__ float As[BM * BK];
     __shared__ float Bs[BM * BK];

     A += cRow * BK * K;
     B += cCol * BK;
     C += cRow * BM * N + cCol * BN;


    const uint inColA = threadIdx.x % BK;
    const uint inRowA = threadIdx.x / BK; 
     
    const uint inRowB = threadIdx.x / BK;
    const uint inColB = threadIdx.x % BK;
    
    float tRes[TM] = {0.0};
     
     for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
         As[inRowA * BK + inColA] = A[inRowA * K + inColA];
         Bs[inRowA * BK + inColA] = B[inRowB * N + inColB];

         A += BK;
         B += BK * N;

