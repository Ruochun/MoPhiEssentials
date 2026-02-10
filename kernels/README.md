# Kernels

This directory contains computational kernels that can run on both CPU and GPU.

## Files to be Added from MoPhi

Please copy the following from the MoPhi repository's `kernels` directory:
- Vector operation kernels
- Matrix operation kernels
- Reduction kernels
- CUDA kernel implementations (*.cu, *.cuh)
- CPU fallback implementations
- All related files

**Note**: Exclude any jitify-related kernels as specified in the requirements.

## Current Placeholder Files

- `vector_ops.h`: Vector operation kernels (placeholder)

These are placeholders and should be replaced with actual implementations from MoPhi.

## CUDA Support

Kernels can be compiled with CUDA support by setting the CMake option:
```bash
cmake -DMOPHI_ENABLE_CUDA=ON ..
```
