#ifndef MOPHI_COMMON_TYPES_H
#define MOPHI_COMMON_TYPES_H

#include <cstdint>
#include <cstddef>

namespace MoPhi {
namespace Common {

// Common type definitions
using Index = std::size_t;
using Int = std::int32_t;
using Real = double;

// Device type enumeration
enum class DeviceType {
    CPU,
    GPU
};

// Memory location
enum class MemoryLocation {
    Host,
    Device,
    Unified
};

} // namespace Common
} // namespace MoPhi

#endif // MOPHI_COMMON_TYPES_H
