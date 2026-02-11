//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_DEFINES_HPP
#define MOPHI_DEFINES_HPP

// #include <limits>
#include <algorithm>
#include <cmath>

#include <common/VariableTypes.hpp>
#include <core/Real3.hpp>
#include "cuda_runtime.h"

namespace mophi {

// =============================================================================
// Structs and consts in this file could be used both on host and device,
// so the simpler (no dependencies), the better.
// =============================================================================

#ifndef MOPHI_HD
    #ifdef __CUDACC__
        #define MOPHI_HD __host__ __device__
    #else
        #define MOPHI_HD
    #endif
#endif

#define MOPHI_MIN(a, b) ((a < b) ? a : b)
#define MOPHI_MAX(a, b) ((a > b) ? a : b)

// Macro constants
#define MOPHI_BITS_PER_BYTE 8u

// Typically is compressed XYZ (Note: if any single axis uses more than ~53 bits, the double-based mapping loses
// integer-exactness)
struct alignas(16) CompLinear3D_128Bit {
    static constexpr unsigned kBitsX = 42;
    static constexpr unsigned kBitsY = 42;
    static constexpr unsigned kBitsZ = 42;

    uint64_t x : kBitsX;
    uint64_t y : kBitsY;
    uint64_t z : kBitsZ;
};

// Typically is compressed vel: 64 total bits: 24 for log-magnitude, 20+20 for oct direction; should be compact & fast
struct alignas(8) CompLog3D_64Bit {
    static constexpr unsigned kBitsMag = 24;  // magnitude code
    static constexpr unsigned kBitsU = 20;    // oct-encoded u in [0,1]
    static constexpr unsigned kBitsV = 20;    // oct-encoded v in [0,1]

    uint64_t u : kBitsU;
    uint64_t v : kBitsV;
    uint64_t m : kBitsMag;
};

// Mesh soup ingredients
struct TetTopo {
    nodeID_t v[4];
};
struct HexTopo {
    nodeID_t v[8];
};

// TetEdgesLocal stores the local edge ids (0..5) of a tet's edges
// Local means this local partition, not 0, 1, 2...
struct TetEdgesLocal {
    uNodeID_t e[6];
};  // (01,02,03,12,13,23) local edge ids

// Verbosity
const verbosity_t VERBOSITY_QUIET = 0;
const verbosity_t VERBOSITY_ERROR = 1;
const verbosity_t VERBOSITY_WARNING = 2;
const verbosity_t VERBOSITY_INFO = 3;
const verbosity_t VERBOSITY_DEBUG = 4;
const verbosity_t VERBOSITY_STEP_DEBUG = 5;

// Facts about elements
constexpr uNodeID_t NODES_PER_TET = 4;
constexpr uNodeID_t EDGES_PER_TET = 6;
constexpr uNodeID_t FACES_PER_TET = 4;
constexpr uNodeID_t NODES_PER_HEX = 8;
constexpr uNodeID_t EDGES_PER_HEX = 12;
constexpr uNodeID_t FACES_PER_HEX = 6;

// Names
const std::string UNSPECIFIED_NAME = "mophi_auto_decide";
const std::string RESERVED_VEL_FS_NAME = "mophi_cfd_vel_fs_default";
const std::string RESERVED_VEL_VAR_NAME = "mophi_cfd_vel_var_default";

}  // namespace mophi

#endif
