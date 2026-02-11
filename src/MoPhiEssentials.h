//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file MoPhiEssentials.h
 * @brief One-stop include header for MoPhiEssentials CPU-side public API
 *
 * This header provides convenient access to all CPU-compatible classes, functions,
 * and utilities in the MoPhiEssentials library. For GPU/CUDA-specific functionality
 * that requires nvcc compilation, use MoPhiEssentialsGPU.cuh instead.
 *
 * @code{.cpp}
 * #include <MoPhiEssentials.h>
 *
 * int main() {
 *     mophi::Real3f vec(1.0f, 2.0f, 3.0f);
 *     mophi::DeviceArray<float> arr(1000);
 *     mophi::GpuManager gpuMgr;
 *     // ...
 * }
 * @endcode
 * 
 * @note For .cu files that need GPU kernels and device code, also include:
 * @code{.cu}
 * #include <MoPhiEssentials.h>
 * #include <MoPhiEssentialsGPU.cuh>
 * @endcode
 */

#ifndef MOPHI_ESSENTIALS_H
#define MOPHI_ESSENTIALS_H

// ============================================================================
// Core API Version and Configuration
// ============================================================================
#include <core/ApiVersion.h>

// ============================================================================
// Foundation: Base Classes and Type Definitions
// ============================================================================
#include <core/BaseClasses.hpp>
#include <common/VariableTypes.hpp>

// ============================================================================
// Core Math and Vector Operations
// ============================================================================
#include <core/Real3.hpp>

// ============================================================================
// Common Definitions and Macros
// ============================================================================
#include <common/Defines.hpp>

// ============================================================================
// Memory Management
// ============================================================================
#include <core/CudaAllocator.hpp>
#include <core/ManagedMemory.hpp>

// ============================================================================
// Logging and Error Handling
// ============================================================================
#include <core/Logger.hpp>

// ============================================================================
// Data Classes and Migration
// ============================================================================
#include <core/DataMigrationHelper.hpp>
#include <core/DataClasses.hpp>

// ============================================================================
// GPU Management and Threading
// ============================================================================
#include <core/GpuManager.h>
#include <core/ThreadManager.hpp>

// ============================================================================
// Common Data Structures
// ============================================================================
#include <common/Compression.hpp>
#include <common/SharedStructs.hpp>
#include <common/Mesh.hpp>

// ============================================================================
// Static Device Subroutines
// ============================================================================
#include <algorithms/StaticDeviceSubroutines.h>

// ============================================================================
// High-Level Utilities
// ============================================================================
#include <core/WavefrontMeshLoader.hpp>
#include <utils/Timer.hpp>
#include <utils/HostHelpers.hpp>
#include <utils/Csv.hpp>
#include <utils/MeshIO.hpp>

#endif  // MOPHI_ESSENTIALS_H
