//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_COMPRESSION_HPP
#define MOPHI_COMPRESSION_HPP

#include <string>
#include <vector>

#include <common/Defines.hpp>
#include <core/DataMigrationHelper.hpp>
#include <core/Real3.hpp>
#include <kernels/Compression.cuh>

namespace mophi {

// ----- Real3 detector -----
template <typename>
struct is_real3 : std::false_type {};
template <typename S>
struct is_real3<Real3<S>> : std::true_type {};
template <typename T>
inline constexpr bool is_real3_v = is_real3<T>::value;

// Extract scalar type if T = Real3<S> (void otherwise)
template <typename T>
struct real3_scalar {
    using type = void;
};
template <typename S>
struct real3_scalar<Real3<S>> {
    using type = S;
};
template <typename T>
using real3_scalar_t = typename real3_scalar<T>::type;

// Float/double 3D vectors detection
template <typename T>
inline constexpr bool is_real3_float_v = is_real3_v<T> && (std::is_same_v<real3_scalar_t<T>, float> ||
                                                           std::is_same_v<real3_scalar_t<T>, double>);

// Data compression type
enum class CompressionKind : uint8_t { None, Log3D_64Bit, Linear3D_128Bit };

template <typename T>
constexpr bool CompressionSupportedFor(CompressionKind k) {
    switch (k) {
        case CompressionKind::None:
            return true;
        case CompressionKind::Linear3D_128Bit:
        case CompressionKind::Log3D_64Bit:
            // For now we only support Real3<float> / Real3<double>
            return is_real3_float_v<T>;
        default:
            return false;
    }
}

template <typename T>
inline void compress_xyz_to_voxel(const std::vector<Real3<T>>& points,
                                  CompLinear3D_128Bit* voxel,
                                  size_t n_points,
                                  const Real3d& LBF,
                                  const Real3d& size) {
    // Do this on host
    for (size_t i = 0; i < n_points; ++i) {
        CompressPoint_T<CompLinear3D_128Bit, T>(&points[i], LBF, size, &voxel[i]);
    }
}

template <typename T>
inline void decompress_voxel_to_xyz(const CompLinear3D_128Bit* voxel,
                                    std::vector<Real3<T>>& points,
                                    size_t n_points,
                                    const Real3d& LBF,
                                    const Real3d& size) {
    // Do this on host
    for (size_t i = 0; i < n_points; ++i) {
        Real3<T> p;
        DecompressPoint_T<CompLinear3D_128Bit, T>(&voxel[i], LBF, size, &p);
        points[i] = p;
    }
}

template <typename T>
inline void compress_vel_to_logscale(const std::vector<Real3<T>>& velocities,
                                     CompLog3D_64Bit* vel_logscale,
                                     size_t n_points,
                                     double minVel,
                                     double maxVel,
                                     double velZeroEps) {
    // Do this on host
    for (size_t i = 0; i < n_points; ++i) {
        CompressLogscale_T<CompLog3D_64Bit, T>(&velocities[i], minVel, maxVel, velZeroEps, &vel_logscale[i]);
    }
}

template <typename T>
inline void decompress_logscale_to_vel(const CompLog3D_64Bit* vel_logscale,
                                       std::vector<Real3<T>>& velocities,
                                       size_t n_points,
                                       double minVel,
                                       double maxVel,
                                       double velZeroEps) {
    // Do this on host
    for (size_t i = 0; i < n_points; ++i) {
        Real3<T> v;
        DecompressLogscale_T<CompLog3D_64Bit, T>(&vel_logscale[i], minVel, maxVel, &v);
        velocities[i] = v;
    }
}

}  // namespace mophi

#endif
