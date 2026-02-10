# MoPhiEssentials

A collection of low-level tools and utilities for multi-physics simulations. This library provides essential infrastructure that can be incorporated as a 3rd party module by other projects, making them "MoPhi-Approved" through standardized data management and communication interfaces.

## Overview

MoPhiEssentials contains the fundamental building blocks for multi-physics solvers:

- **CPU-GPU Vectors**: Unified data structures for computation on both CPU and GPU
- **Mesh Format**: Standardized mesh representation for various solver types
- **Core Utilities**: Low-level infrastructure for data management and communication
- **Algorithms**: Essential algorithms for multi-physics simulations
- **Kernels**: Computational kernels optimized for performance

## Project Structure

```
MoPhiEssentials/
├── algorithms/     # Low-level algorithms
├── core/          # Core functionality and data structures
├── kernels/       # Computational kernels (CPU/GPU)
├── common/        # Common utilities and definitions
├── utils/         # Utility functions and helpers
├── tests/         # Unit and integration tests
├── examples/      # Example applications and demos
└── docs/          # Documentation
```

## Purpose

This repository is designed to be easily integrated into other projects as a dependency. By using MoPhiEssentials, projects gain:

1. **Standardized Infrastructure**: Common data structures and management classes
2. **Interoperability**: Easy communication between different solver components
3. **Performance**: Optimized implementations for both CPU and GPU
4. **MoPhi Compatibility**: Seamless integration with the MoPhi ecosystem

## Usage as a 3rd Party Module

### CMake Integration

```cmake
# In your CMakeLists.txt
add_subdirectory(external/MoPhiEssentials)
target_link_libraries(your_project PRIVATE MoPhiEssentials)
```

### Git Submodule

```bash
git submodule add https://github.com/Ruochun/MoPhiEssentials external/MoPhiEssentials
git submodule update --init --recursive
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Testing

```bash
cd build
ctest
```

## Requirements

- C++17 or later
- CMake 3.15 or later
- CUDA Toolkit (optional, for GPU support)

## License

BSD 3-Clause License. See LICENSE file for details.

## Contributing

This is a private repository. Please contact the maintainers for contribution guidelines.
