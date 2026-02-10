# Documentation

This directory contains documentation for MoPhiEssentials.

## Overview

MoPhiEssentials is a library of low-level tools and utilities for multi-physics simulations.

## Components

### Common
- `types.h`: Common type definitions (Index, Real, etc.)
- `macros.h`: Compiler macros and version information

### Core
- `vector.h`: CPU-GPU unified vector class
- `mesh.h`: Mesh data structure for finite element/volume methods

### Algorithms
- `sorting.h`: Sorting algorithms
- `linear_algebra.h`: Basic linear algebra operations

### Kernels
- `vector_ops.h`: Computational kernels for vector operations

### Utils
- `logger.h`: Logging utility
- `timer.h`: Timing utility

## Building Documentation

Documentation can be built using Doxygen (if available):

```bash
cd docs
doxygen Doxyfile
```

## API Reference

Coming soon. The API will be documented using Doxygen comments in the header files.

## Integration Guide

See the main README.md for information on integrating MoPhiEssentials into your project.
