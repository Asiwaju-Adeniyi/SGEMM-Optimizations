
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


template <int BLOCKSIZE>
__global__ void c_GEMM(int M, int N, int K, float alpha,
                       const float* A, const float* B,
                       float beta, float* C)
{
    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;

        for (int k = 0; k < K; ++k) {
            val += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = alpha * val + beta * C[row * N + col];
    }
}


int main() {

    int M = 1024;
    int N = 1024;
    int K = 1024;

    float alpha = 1.0f;
    float beta  = 0.0f;

    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N]; 
    float *h_C = new float[M * N];

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);

    cudaEvent_t startEvent, endEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent, 0);

    c_GEMM<32><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);

    float duration = 0.0f;

    cudaEventElapsedTime(&duration, startEvent, endEvent);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

        for (int i = 0; i < sizeC; i++) {
        std::cout << "C[" << i << "]" << h_C[i] << std::endl;
    }

    std::cout << duration << "ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C; 


}
