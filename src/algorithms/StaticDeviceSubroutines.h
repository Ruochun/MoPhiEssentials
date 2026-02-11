//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_STATIC_DEVICE_SUBROUTINES_H
#define MOPHI_STATIC_DEVICE_SUBROUTINES_H

#include <common/Defines.hpp>
#include <common/Compression.hpp>
#include <common/SharedStructs.hpp>
#include <common/Mesh.hpp>

#include <core/GpuManager.h>
#include <core/CudaAllocator.hpp>
#include <core/DataClasses.hpp>

namespace mophi {

// ========================================================================
// Typically, these are device-side relatively heavy-duty subroutines.
// Other files of this project may link against this .h file, without knowing
// it's in fact about CUDA, as they are C++ only.
// Functions here are statically compiled against available CUDA versions,
// unlike other just-in-time compiled kernels you see in other files of
// this project.
// ========================================================================

////////////////////////////////////////////////////////////////////////////////
// Dynamics thread kernels
////////////////////////////////////////////////////////////////////////////////

void launch_xyz_to_voxel_kernel(const Real3d* LBF,
                                const Real3d* size,
                                const Real3d* xyz,
                                CompLinear3D_128Bit* voxel_pos,
                                size_t n,
                                cudaStream_t& stream);

void launch_laplace_assemble(const TetTopo* tets,
                             const TetEdgesLocal* tetEdges,
                             const CompLinear3D_128Bit* voxel_pos,
                             const CompLog3D_64Bit* vels,
                             const SolverParams* P,
                             const DomainInfo* domain,
                             COOScatterView<uNodeID_t, storeData_t> out,
                             size_t n,
                             cudaStream_t& stream);
}  // namespace mophi

#endif
