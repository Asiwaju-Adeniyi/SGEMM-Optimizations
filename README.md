# CUDA MatMul Worklog — Reimplementing Simon Bøhm’s Optimized Kernels

This repo contains my reimplementation of the CUDA matrix multiplication kernels from  
**Simon Bøhm’s “How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog.”**

The goal of this project is *not* to rewrite Simon’s theory or blog, but to recreate the **iterative kernel sequence** — from naive matmul to highly optimized SGEMM — purely through code.

## What’s Included
This repository mirrors Simon’s worklog by implementing each kernel stage:

- **Naive CUDA matmul**
- **Coalesced global memory access**
- **Shared memory tiling**
- **Blocking & register tiling**
- **Double-buffered shared memory**
- **Warp-level tiling**
- **Vectorized memory loads**
- **Occupancy & ILP improvements**
- **Final near-cuBLAS performance kernel**

Each version is kept separate and benchmarked so you can see the performance progression step by step.

## Scope of the Repo
- **Only kernels + benchmarking code**  
- No architecture explanations  
- No rewritten theory (refer to [Simon’s blog](https://siboehm.com/articles/22/CUDA-MMM) for that)

This repo exists purely as a **learning-through-reimplementation playground**.

## Benchmarks
Each kernel includes a lightweight benchmark (inspired by wangzyn’s setup, same as Simon’s).  
Results will vary by GPU, compiler version, and block size.

## Reference
Original blog:  
*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog* by Simon Bøhm (Dec 2022)

All concepts, ideas, and kernel progression come from Simon’s work — these implementations are my own recreations for practice and understanding.

## Acknowledgements
Thanks to Simon Bøhm for the phenomenal worklog and for sharing the entire journey from naive matmul to near-cuBLAS performance. This repo follows the same path through hands-on code.
