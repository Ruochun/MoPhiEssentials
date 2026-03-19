//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <cub/cub.cuh>
#include "StaticDeviceSubroutines.h"

#include "../core/Logger.hpp"
#include "../core/DataClasses.hpp"
#include "CubWrappers.cuh"

namespace mophi {

// ========================================================================
// Instantiations of CUB-based subroutines
// These functions interconnecting the cub-part and cpp-part of the code cannot be fully templated...
// Instead, all possible uses must be explicitly instantiated.
// ========================================================================

}  // namespace mophi
