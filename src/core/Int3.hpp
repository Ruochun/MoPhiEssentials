//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file Int3.hpp
 * @brief Integer 3-component vector type mophi::Int3.
 *
 * Int3 is a type alias for mophi::Real3<int>.  It carries the complete Real3
 * interface (arithmetic operators, Clamp, Abs, etc.) with integer semantics.
 * Note that floating-point operations such as Floor, Frac, Fmod, Lerp, Normalize,
 * and Length are mathematically valid only for floating-point element types;
 * calling them on Int3 compiles but may yield unexpected results.
 *
 * Usage:
 * @code{.cpp}
 * #include "core/Int3.hpp"
 *
 * mophi::Int3 idx(1, 2, 3);
 * mophi::Int3 clamped = idx.Clamp(0, 5);
 * mophi::Int3 abs_idx = idx.Abs();
 * @endcode
 */

#ifndef MOPHI_INT3_HPP
#define MOPHI_INT3_HPP

#include "Real3.hpp"

namespace mophi {

/// Integer 3-component vector (alias for Real3<int>).
using Int3 = Real3<int>;

}  // namespace mophi

#endif  // MOPHI_INT3_HPP
