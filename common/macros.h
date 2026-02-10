#ifndef MOPHI_COMMON_MACROS_H
#define MOPHI_COMMON_MACROS_H

// Compiler-specific macros
#ifdef __CUDACC__
    #define MOPHI_HOST __host__
    #define MOPHI_DEVICE __device__
    #define MOPHI_HOST_DEVICE __host__ __device__
    #define MOPHI_GLOBAL __global__
#else
    #define MOPHI_HOST
    #define MOPHI_DEVICE
    #define MOPHI_HOST_DEVICE
    #define MOPHI_GLOBAL
#endif

// Utility macros
#define MOPHI_UNUSED(x) (void)(x)

// Version information
// Note: These must be kept in sync with CMakeLists.txt
#define MOPHI_VERSION_MAJOR 0
#define MOPHI_VERSION_MINOR 1
#define MOPHI_VERSION_PATCH 0

#endif // MOPHI_COMMON_MACROS_H
