# Migration Guide: MoPhi to MoPhiEssentials

This document guides the process of copying the low-level components from MoPhi to MoPhiEssentials.

## Overview

MoPhiEssentials is a new repository that extracts the low-level infrastructure from MoPhi, allowing it to be used as a standalone 3rd party library.

## What to Copy

### 1. Common Directory (`common/`)
Copy all files from `MoPhi/common/` to `MoPhiEssentials/common/`:
- Type definitions
- Macros and constants
- Shared enumerations
- Version information

### 2. Core Directory (`core/`)
Copy all files from `MoPhi/core/` to `MoPhiEssentials/core/`:
- CPU-GPU vector implementations
- Mesh data structures
- Memory management classes
- Data container classes
- Any low-level data structures

### 3. Algorithms Directory (`algorithms/`)
Copy algorithm implementations from `MoPhi/algorithms/` to `MoPhiEssentials/algorithms/`:
- Sorting algorithms
- Linear algebra operations
- Search algorithms
- Graph algorithms
- Mesh processing algorithms

**EXCLUDE**: Any jitify-related algorithms

### 4. Kernels Directory (`kernels/`)
Copy kernel implementations from `MoPhi/kernels/` to `MoPhiEssentials/kernels/`:
- CUDA kernels (*.cu, *.cuh files)
- CPU kernel implementations
- Vector operation kernels
- Matrix operation kernels
- Reduction kernels

**EXCLUDE**: Any jitify-related kernels

### 5. Utils Directory (`utils/`)
Copy utility functions from `MoPhi/utils/` to `MoPhiEssentials/utils/`:
- Logging utilities
- Timing/profiling utilities
- File I/O utilities
- String manipulation
- Error handling

### 6. Tests
Copy relevant tests from `MoPhi/tests/` to `MoPhiEssentials/tests/`:
- Tests for vector operations
- Tests for mesh operations
- Tests for algorithms
- Tests for kernels
- Tests for utilities

Only copy tests that relate to the low-level components being migrated.

### 7. Examples/Demos
Copy supportive demos from `MoPhi/examples/` or `MoPhi/demos/` to `MoPhiEssentials/examples/`:
- Vector usage examples
- Mesh creation examples
- Algorithm demonstrations
- Kernel benchmarks

Only copy examples that demonstrate the low-level infrastructure.

## What NOT to Copy

### High-Level Components
Do NOT copy:
- Multi-physics solver implementations
- Python wrapper code
- Solver coupling infrastructure
- High-level simulation drivers
- Any jitify-related code

### Identifying Jitify-Related Code
Look for and exclude:
- Files with "jitify" in the name
- Code that uses `jitify::` namespace
- JIT compilation infrastructure
- Runtime kernel compilation code

## Migration Steps

### Step 1: Setup
```bash
cd MoPhiEssentials
git checkout -b feature/migrate-from-mophi
```

### Step 2: Copy Directory by Directory
For each directory (common, core, algorithms, kernels, utils):

1. Copy all relevant files from MoPhi
2. Remove jitify-related files
3. Update include paths if necessary
4. Verify CMakeLists.txt includes new files
5. Test compilation

### Step 3: Update Include Paths
If files in MoPhi had includes like:
```cpp
#include "high_level/solver.h"
```

Remove those includes if they reference high-level components not in MoPhiEssentials.

### Step 4: Update CMake
Ensure all new source files are properly included in the respective CMakeLists.txt files.

### Step 5: Build and Test
```bash
mkdir build && cd build
cmake ..
make
ctest
```

### Step 6: Verify Examples
Run the examples to ensure they work:
```bash
./examples/example_vector
./examples/example_mesh
./examples/example_algorithms
```

## Verification Checklist

- [ ] All common utilities are copied
- [ ] CPU-GPU vector implementation is present
- [ ] Mesh data structure is complete
- [ ] Algorithms compile and work
- [ ] Kernels compile (both CPU and CUDA)
- [ ] Utilities are functional
- [ ] No jitify-related code is included
- [ ] No high-level solver code is included
- [ ] Tests pass
- [ ] Examples run successfully
- [ ] Documentation is updated

## Post-Migration

After migration:
1. Update version numbers
2. Create a release tag
3. Update README with accurate information
4. Document the API
5. Create integration examples for downstream projects

## Need Help?

If the MoPhi repository structure differs from what's described here, please adjust the migration accordingly. The key principle is: **copy only the low-level infrastructure, exclude high-level solvers and jitify.**
