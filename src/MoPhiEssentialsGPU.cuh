//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file MoPhiEssentialsGPU.cuh
 * @brief GPU/CUDA-specific headers for MoPhiEssentials
 *
 * This header includes all CUDA device code and kernel utilities that require
 * nvcc (NVIDIA CUDA Compiler) for compilation. Use this header in .cu files
 * that need GPU functionality.
 *
 * For CPU-only code, use MoPhiEssentials.h instead.
 *
 * @code{.cu}
 * #include <MoPhiEssentials.h>      // CPU-side functionality
 * #include <MoPhiEssentialsGPU.cuh> // GPU-side functionality
 *
 * __global__ void myKernel() {
 *     // Use GPU utilities here
 * }
 * @endcode
 */

#ifndef MOPHI_ESSENTIALS_GPU_CUH
#define MOPHI_ESSENTIALS_GPU_CUH

// ============================================================================
// Kernel Utilities (Device Code) - Requires nvcc
// ============================================================================
#include "kernels/Compression.cuh"
#include "kernels/HelperKernels.cuh"
#include "kernels/CUDAMathHelpers.cuh"

// ============================================================================
// Algorithm Utilities (Device Code) - Requires nvcc
// ============================================================================
#include "algorithms/Utilities.cuh"
#include "algorithms/CubWrappers.cuh"

#endif  // MOPHI_ESSENTIALS_GPU_CUH
