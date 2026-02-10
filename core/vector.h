#ifndef MOPHI_CORE_VECTOR_H
#define MOPHI_CORE_VECTOR_H

#include "common/types.h"
#include "common/macros.h"
#include <vector>
#include <memory>

namespace MoPhi {
namespace Core {

/**
 * @brief Unified vector class that can work on both CPU and GPU
 * 
 * This is a placeholder for the actual implementation from MoPhi.
 * The actual implementation should be copied from the MoPhi repository.
 */
template<typename T>
class Vector {
public:
    using value_type = T;
    using size_type = Common::Index;

    // Constructors
    Vector() : size_(0), capacity_(0), location_(Common::MemoryLocation::Host) {}
    explicit Vector(size_type size, Common::MemoryLocation loc = Common::MemoryLocation::Host);
    
    // Copy and move operations
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;
    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other) noexcept;
    
    // Destructor
    ~Vector();

    // Size operations
    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    void resize(size_type new_size);
    void reserve(size_type new_capacity);
    void clear();

    // Data access
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    // Memory operations
    void copyToDevice();
    void copyToHost();
    Common::MemoryLocation location() const { return location_; }

private:
    T* data_;
    size_type size_;
    size_type capacity_;
    Common::MemoryLocation location_;
};

} // namespace Core
} // namespace MoPhi

#endif // MOPHI_CORE_VECTOR_H
