//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

// Backward-compatibility wrapper: includes all data container headers.
// Prefer including the specific header directly:
//   - DataContainerBase.hpp  for DataContainer and RotatingDataContainer
//   - DataClassesCpu.hpp     for HostArrayContainer and HostArrayRotatingPool
//   - DataClassesCuda.hpp    for DualArrayContainer and DeviceArrayRotatingPool (CUDA only)

#ifndef MOPHI_DATA_CLASSES_HPP
#define MOPHI_DATA_CLASSES_HPP

#include <core/DataContainerBase.hpp>
#include <core/DataClassesCpu.hpp>
#ifdef MOPHI_USE_CUDA
    #include <core/DataClassesCuda.hpp>
#endif

#endif
