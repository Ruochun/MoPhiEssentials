# MoPhiEssentials

Low-level infrastructure for building MoPhi-compatible physics solvers.

## Overview

MoPhiEssentials provides the foundational utilities that make a project "MoPhi-Approved". By using these common low-level tools, different physics solvers can easily communicate and be combined at a high level within the MoPhi framework.

## What's Included

### Core Infrastructure (`src/core/`)
- **Memory Management**
  - `CudaAllocator.hpp` - Unified memory allocators
  - `ManagedMemory.hpp` - CPU-GPU managed memory
  - `DataClasses.hpp` - Device arrays and data structures
  - `DataMigrationHelper.hpp` - Host-device data synchronization

- **GPU Management**
  - `GpuManager.h/cpp` - Multi-GPU and stream management
  - GPU device selection and resource allocation

- **Math Utilities**
  - `Real3.hpp` - Template vector classes (Real3<float>, Real3<double>)
  - CPU and GPU compatible vector operations

- **Utilities**
  - `Logger.hpp` - Thread-safe logging and error handling
  - `ThreadManager.hpp` - Thread pool management
  - `Paths.h/cpp` - File path utilities
  - `WavefrontMeshLoader.hpp` - Mesh loading utilities

### Algorithms (`src/algorithms/`)
- **CUB Wrappers** (`CubWrappers.cuh`) - Convenience wrappers for NVIDIA CUB library operations
- **Static Device Subroutines** - Pre-compiled CUDA kernels for common operations
- **Compression Utilities** - Data compression for GPU memory optimization

### Device Kernels (`src/kernels/`)
- `Compression.cuh` - Quantization and octahedral encoding
- `HelperKernels.cuh` - Common device-side math helpers
- `CUDAMathHelpers.cuh` - CUDA vector math extensions
- `Constants.cuh` - Device constants

### Common Definitions (`src/common/`)
- `Defines.hpp` - Common macros and compile-time definitions
- `VariableTypes.hpp` - Type definitions
- `Compression.hpp` - Compression data structures
- `SharedStructs.hpp` - Shared data structures for CPU-GPU
- `Mesh.hpp` - Mesh data structures

## Why MoPhiEssentials?

When multiple physics solvers use MoPhiEssentials, they gain:

1. **Common Data Structures** - Vectors, arrays, meshes that work seamlessly on CPU and GPU
2. **Memory Interoperability** - Same memory management patterns across all solvers
3. **GPU Infrastructure** - Consistent device and stream management
4. **Easy Integration** - Solvers can communicate without data conversion overhead

This makes it trivial to combine different physics solvers (CFD, FEA, DEM, etc.) into multi-physics simulations.

## Building

### Prerequisites
- CMake 3.18 or newer
- CUDA Toolkit (with CUB library)
- C++17 or newer compiler
- Linux, Windows, or macOS

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/Ruochun/MoPhiEssentials.git
cd MoPhiEssentials

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make

# Optional: Install system-wide
sudo make install
```

### Build Options

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Specify C++ standard
cmake -DTargetCXXStandard=STD_CXX17 ..

# Specify CUDA architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86" ..
```

## Using MoPhiEssentials in Your Project

### Option 1: CMake Subdirectory

```cmake
# In your CMakeLists.txt
add_subdirectory(path/to/MoPhiEssentials)
target_link_libraries(your_target PUBLIC mophi_essentials)
```

### Option 2: Installed Package

```bash
# Install MoPhiEssentials
cd MoPhiEssentials/build
sudo make install
```

```cmake
# In your CMakeLists.txt
find_package(MoPhiEssentials REQUIRED)
target_link_libraries(your_target PUBLIC MoPhiEssentials::mophi_essentials)
```

### Option 3: Git Submodule

```bash
# Add as submodule
git submodule add https://github.com/Ruochun/MoPhiEssentials.git external/MoPhiEssentials
git submodule update --init --recursive
```

```cmake
# In your CMakeLists.txt
add_subdirectory(external/MoPhiEssentials)
target_link_libraries(your_target PUBLIC mophi_essentials)
```

## Example Usage

### Using Real3 Vectors

```cpp
#include <core/Real3.hpp>

mophi::Real3f a(1.0f, 2.0f, 3.0f);
mophi::Real3f b(4.0f, 5.0f, 6.0f);

// Vector operations
mophi::Real3f c = a + b;
mophi::Real3f d = a.Cross(b);
float dot = a ^ b;  // dot product
float len = a.Length();
```

### Using Device Arrays

```cpp
#include <core/DataClasses.hpp>

mophi::DeviceArray<float> gpu_array(1000);
gpu_array.resize(2000);
gpu_array.SetVal(3.14f, 5);  // Set value at index 5
```

### Using Dual Arrays (CPU-GPU)

```cpp
#include <core/DataMigrationHelper.hpp>

mophi::DualArray<float> dual_array(1000, 1.0f);

// Modify on host
dual_array[5] = 3.14f;

// Sync to device
dual_array.ToDevice();

// Kernel uses device data...

// Sync back to host
dual_array.ToHost();
```

### Using GPU Manager

```cpp
#include <core/GpuManager.h>

mophi::GpuManager gpuMgr(4, {0, 1});  // 4 streams across GPUs 0 and 1

auto stream = gpuMgr.GetAvailableStream();
// Use stream.stream for CUDA operations
gpuMgr.SetStreamAvailable(stream);
```

## Demo Programs

The `demo/` directory contains example programs:

- `HelloWorld.cpp` - Basic timer usage
- `TestMsg.cpp` - Logger demonstration
- `TestReal3.cu` - Vector operations on CPU and GPU
- `TestDualArray.cpp` - CPU-GPU data synchronization
- `TestContainers.cpp` - Device array pool management

Build and run demos:
```bash
cd build/bin
./TestReal3
./TestMsg
```

## API Documentation

### Key Classes

- `mophi::Real3<T>` - 3D vector template class
- `mophi::DeviceArray<T>` - GPU-only array
- `mophi::DualArray<T>` - CPU-GPU synchronized array
- `mophi::DualStruct<T>` - CPU-GPU synchronized struct
- `mophi::GpuManager` - Multi-GPU stream manager
- `mophi::Logger` - Thread-safe logging system

### Key Macros

- `MOPHI_HD` - Mark functions for host and device
- `MOPHI_GPU_CALL()` - Check CUDA errors
- `MOPHI_INFO()`, `MOPHI_WARNING()`, `MOPHI_ERROR()` - Logging macros

## Contributing

Please see `CONTRIBUTORS.md` for contribution guidelines.

## License

Copyright (c) 2025, Ruochun Zhang  
SPDX-License-Identifier: BSD-3-Clause

See `LICENSE` file for full license text.

## Citation

If you use MoPhiEssentials in your research, please cite:

```bibtex
@software{mophi_essentials,
  title = {MoPhiEssentials: Low-level Infrastructure for Multi-Physics Solvers},
  author = {Zhang, Ruochun},
  year = {2025},
  url = {https://github.com/Ruochun/MoPhiEssentials}
}
```

## Contact

- **Author**: Ruochun Zhang
- **Repository**: https://github.com/Ruochun/MoPhiEssentials
- **Issues**: https://github.com/Ruochun/MoPhiEssentials/issues