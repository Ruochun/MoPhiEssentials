//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_VAR_TYPES
#define MOPHI_VAR_TYPES

#include <stdint.h>

namespace mophi {

// static_assert(sizeof(size_t) >= sizeof(unsigned long long), "This code should be compiled on 64-bit systems!");

#if defined(_WIN64)
typedef long long ssize_t;
#elif defined(_WIN32)
typedef long ssize_t;
#endif

typedef int verbosity_t;  ///< Verbosity type, used for logging and debugging

typedef int nodeID_t;                  ///< Node ID type, used for indexing nodes in the mesh
typedef unsigned int uNodeID_t;        ///< Unsigned node ID type
typedef int nodeIDGlobal_t;            ///< Node ID global type, usually the same as nodeID_t
typedef unsigned int uNodeIDGlobal_t;  ///< Unsigned global node ID type

typedef int8_t meshTag_t;    ///< Mesh tag type, used for tagging mesh elements
typedef int16_t meshPart_t;  ///< Mesh partition number type

typedef double nodeCoord_t;  ///< Node coordinate type, used for node positions INSIDE KERNELS
typedef float storeData_t;   ///< General intermediate data's type

typedef float usrCFD_t;  ///< User-facing CFD scalar type (u, p, etc.)

}  // namespace mophi

#endif
