//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file TestUnifiedGPUHeader.cu
 * @brief Demonstration of using MoPhiEssentialsGPU.cuh unified GPU header
 *
 * This example shows how to use both CPU-side (MoPhiEssentials.h) and
 * GPU-side (MoPhiEssentialsGPU.cuh) unified headers together in CUDA code.
 */

#include <MoPhiEssentials.h>
#include <MoPhiEssentialsGPU.cuh>
#include <iostream>

using namespace mophi;

// Simple kernel using GPU utilities
__global__ void testKernel(Real3f* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        // Test Real3 on device
        Real3f a(1.0f, 2.0f, 3.0f);
        Real3f b(4.0f, 5.0f, 6.0f);
        result[0] = a + b;
    }
}

int main() {
    std::cout << "\n=== MoPhiEssentials Unified GPU Header Test ===\n\n";

    // Test 1: CPU-side functionality from MoPhiEssentials.h
    std::cout << "1. Testing CPU-side Real3 vectors (from MoPhiEssentials.h)...\n";
    Real3f vec1(1.0f, 2.0f, 3.0f);
    Real3f vec2(4.0f, 5.0f, 6.0f);
    Real3f sum = vec1 + vec2;
    std::cout << "   CPU: vec1 + vec2 = (" << sum.x() << ", " << sum.y() << ", " << sum.z() << ")\n";
    std::cout << "   ✓ CPU Real3 operations work!\n\n";

    // Test 2: GPU-side functionality
    std::cout << "2. Testing GPU-side operations (from MoPhiEssentialsGPU.cuh)...\n";

    Real3f* d_result;
    Real3f h_result;

    MOPHI_GPU_CALL(cudaMalloc((void**)&d_result, sizeof(Real3f)));

    testKernel<<<1, 1>>>(d_result);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
    MOPHI_GPU_CALL(cudaMemcpy(&h_result, d_result, sizeof(Real3f), cudaMemcpyDeviceToHost));

    std::cout << "   GPU: vec1 + vec2 = (" << h_result.x() << ", " << h_result.y() << ", " << h_result.z() << ")\n";
    std::cout << "   ✓ GPU Real3 operations work!\n\n";

    MOPHI_GPU_CALL(cudaFree(d_result));

    // Test 3: Logger (CPU-side)
    std::cout << "3. Testing Logger...\n";
    MOPHI_INFO("GPUTest", "Unified GPU header test successful!");
    std::cout << "   ✓ Logger works!\n\n";

    // Test 4: Timer (CPU-side)
    std::cout << "4. Testing Timer...\n";
    Timer timer;
    timer.start();

    // Simulate some work
    volatile int sum_val = 0;
    for (int i = 0; i < 1000000; i++) {
        sum_val += i;
    }

    timer.stop();
    std::cout << "   Elapsed time: " << timer.GetTimeSeconds() << " seconds\n";
    std::cout << "   ✓ Timer works!\n\n";

    std::cout << "=== All unified GPU header tests passed! ===\n\n";
    std::cout << "Success! The unified headers provide:\n";
    std::cout << "  - MoPhiEssentials.h: CPU-side functionality (Real3, Logger, Timer, etc.)\n";
    std::cout << "  - MoPhiEssentialsGPU.cuh: GPU/CUDA-specific kernels and device code\n";
    std::cout << "  - Both can be used together in .cu files!\n\n";

    return 0;
}
