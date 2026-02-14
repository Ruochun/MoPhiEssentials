//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_DATA_MIGRATION_HPP
#define MOPHI_DATA_MIGRATION_HPP

#include <cassert>
#include <optional>
#include <unordered_map>
#include <core/Logger.hpp>
#include <core/BaseClasses.hpp>
#include <core/CudaAllocator.hpp>
#include <common/VariableTypes.hpp>

namespace mophi {

// A to-device memcpy wrapper
template <typename T>
void cuda_copy_to_device(T* pD, T* pH) {
    MOPHI_GPU_CALL(cudaMemcpy(pD, pH, sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void cuda_copy_to_device(T* pD, T* pH, size_t n) {
    MOPHI_GPU_CALL(cudaMemcpy(pD, pH, n * sizeof(T), cudaMemcpyHostToDevice));
}

// A to-host memcpy wrapper
template <typename T>
void cuda_copy_to_host(T* pH, T* pD) {
    MOPHI_GPU_CALL(cudaMemcpy(pH, pD, sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void cuda_copy_to_host(T* pH, T* pD, size_t n) {
    MOPHI_GPU_CALL(cudaMemcpy(pH, pD, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ptr being a reference to a pointer is crucial
template <typename T>
inline void device_ptr_dealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    MOPHI_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        MOPHI_GPU_CALL(cudaFree(ptr));
}

// You have to deal with it yourself if ptr is an already-used device pointer
template <typename T>
inline void device_ptr_alloc(T*& ptr, size_t size) {
    MOPHI_GPU_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

template <typename T>
inline void host_ptr_dealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    MOPHI_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        MOPHI_GPU_CALL(cudaFreeHost(ptr));
}
template <typename T>
inline void host_ptr_alloc(T*& ptr, size_t size) {
    MOPHI_GPU_CALL(cudaMallocHost((void**)&ptr, size * sizeof(T)));
}

// Managed advise doesn't seem to do anything...
#define MOPHI_ADVISE_DEVICE(vec, device) \
    { advise(vec, ManagedAdvice::PREFERRED_LOC, device); }
#define MOPHI_MIGRATE_TO_DEVICE(vec, device, stream) \
    { migrate(vec, device, stream); }

// MOPHI_DUAL_ARRAY_RESIZE is a reminder for developers that a work array is resized, and this may automatically change
// the external device pointer this array's bound to. Therefore, after this call, syncing the data pointer bundle
// (granData) to device may be needed, and you remember to cudaSetDevice beforehand so it allocates to correct places.
#define MOPHI_DUAL_ARRAY_RESIZE(vec, newsize, val) \
    { vec.resize(newsize, val); }
#define MOPHI_DUAL_ARRAY_RESIZE_NOVAL(vec, newsize) \
    { vec.resize(newsize); }

// Simply a reminder that this is a device array resize, to distinguish from some general .resize calls
#define MOPHI_DEVICE_ARRAY_RESIZE(vec, newsize) \
    { vec.resize(newsize); }

// Use (void) to silence unused warnings.
// #define assertm(exp, msg) assert(((void)msg, exp))

// Used for wrapping host data structures so they become usable on GPU. This class' data should not be modified on the
// device, as there's no mechanism to detect device-side changes.
template <typename T>
class DualStruct : private NonCopyable {
  private:
    T* host_data;           // Pointer to host memory (pinned)
    T* device_data;         // Pointer to device memory
    bool modified_on_host;  // Flag to track if host data has been modified
  public:
    using value_type = T;
    // Constructor: Initialize and allocate memory for both host and device
    DualStruct() : modified_on_host(false) {
        MOPHI_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));
    }

    // Constructor: Initialize and allocate memory for both host and device with init values
    DualStruct(T init_val) : modified_on_host(false) {
        MOPHI_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));

        *host_data = init_val;

        ToDevice();
    }

    // Destructor: Free memory
    ~DualStruct() { free(); }

    // Set the value on host and mark it as modified
    void SetVal(const T& val) {
        *host_data = val;
        MarkModified();
    }

    // Get the value from host (const version)
    const T& GetValue() const { return *host_data; }
    // Get the value from host (non-const version, marks as modified)
    T& GetValue() {
        MarkModified();
        return *host_data;
    }

    // Free both host and device memory
    void free() {
        host_ptr_dealloc(host_data);      // Free pinned memory
        device_ptr_dealloc(device_data);  // Free device memory
        UnmarkModified();
    }

    // Synchronize changes from host to device
    void ToDevice() {
        MOPHI_GPU_CALL(cudaMemcpy(device_data, host_data, sizeof(T), cudaMemcpyHostToDevice));
        UnmarkModified();
    }

    // Synchronize changes from device to host
    void ToHost(bool force = false) {
        if (!force && modified_on_host) {
            MOPHI_ERROR(
                std::string("DualStruct: ToHost called but host data has been modified without syncing to device."));
        }
        MOPHI_GPU_CALL(cudaMemcpy(host_data, device_data, sizeof(T), cudaMemcpyDeviceToHost));
        // Forced or not, the host data is overwritten, so we unmark it
        UnmarkModified();
    }

    // Check if host data has been modified and not synced
    bool CheckNoPendingModification() { return !modified_on_host; }

    void MarkModified() { modified_on_host = true; }

    void UnmarkModified() { modified_on_host = false; }

    // Accessor for host data (using the arrow operator)
    T* operator->() {
        MarkModified();
        return host_data;
    }

    // Accessor for host data (using the arrow operator)
    T* operator->() const { return host_data; }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() {
        MarkModified();
        return *host_data;  // Return a reference to the value
    }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() const {
        return *host_data;  // Return a reference to the value
    }

    // Overloaded operator& for device pointer access
    T* operator&() const {
        ensure_device_data();
        return device_data;  // Return device pointer when using &
    }

    // Getter for the device pointer
    T* device() {
        ensure_device_data();
        return device_data;
    }

    // Getter for the host pointer
    T* host() {
        MarkModified();
        return host_data;
    }

    // Get host or device size in bytes
    size_t GetNumBytes() const { return sizeof(T); }

  private:
    // Ensure device data is up-to-date before accessing it
    void ensure_device_data() {
        if (modified_on_host) {
            ToDevice();
        }
    }
};

// -------------------------------------------------
// CPU--GPU unified array, leveraging pinned memory
// -------------------------------------------------

template <typename T>
class DualArray : private NonCopyable {
  public:
    using PinnedVector = std::vector<T, PinnedAllocator<T>>;
    using value_type = T;

    explicit DualArray() { ensure_host_vector(); }

    DualArray(size_t n) { resize(n); }

    DualArray(size_t n, T val) { resize(n, val); }

    DualArray(const std::vector<T>& vec) { AttachHostVector(vec, /*deep_copy=*/true); }

    ~DualArray() { free(); }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        ResizeHost(n);
        ResizeDevice(n);
    }

    // This resize flavor fills host values only!
    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        ResizeHost(n, val);
        ResizeDevice(n);
    }

    void ResizeHost(size_t n) {
        ensure_host_vector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        update_host_mem_counter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    void ResizeHost(size_t n, const T& val) {
        ensure_host_vector();  // allocates pinned vec if null
        // This method may have meaningful value changes
        pre_host_access();
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        update_host_mem_counter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
    }

    // m_device_capacity is allocated memory, not array usable data range.
    // Also, this method preserves already-existing device data.
    void ResizeDevice(size_t n, bool allow_shrink = false) {
        if (!allow_shrink && m_device_capacity >= n)
            return;

        T* new_device_ptr = nullptr;
        device_ptr_alloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount
        if (m_device_ptr && m_device_capacity > 0) {
            size_t copy_count = std::min(n, m_device_capacity);
            MOPHI_GPU_CALL(cudaMemcpy(new_device_ptr, m_device_ptr, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        // Free old memory and update bookkeeping
        update_device_mem_counter(-(ssize_t)(m_device_capacity * sizeof(T)));
        device_ptr_dealloc(m_device_ptr);

        m_device_ptr = new_device_ptr;
        update_bound_device_pointer();

        update_device_mem_counter(static_cast<ssize_t>(n * sizeof(T)));
        m_device_capacity = n;
    }

    void FreeHost() {
        if (m_host_vec_ptr) {
            update_host_mem_counter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
    }

    void FreeDevice() {
        device_ptr_dealloc(m_device_ptr);
        update_device_mem_counter(-(ssize_t)(m_device_capacity * sizeof(T)));
        m_device_ptr = nullptr;
        m_device_capacity = 0;
        update_bound_device_pointer();
    }

    void free() {
        FreeDevice();
        FreeHost();
        DeclareSync();
    }

    void ToDevice() {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (count > m_device_capacity)
            ResizeDevice(count);
        MOPHI_GPU_CALL(cudaMemcpy(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice));
        // Full sync will make the time stamp the same
        m_device_mod_time = m_host_mod_time;
    }

    void ToDevice(size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Partial flavor aims for speed, no size check
        // When use partial send, users are responsible for managing the time stamp
        MOPHI_GPU_CALL(
            cudaMemcpy(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void ToDeviceAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (count > m_device_capacity)
            ResizeDevice(count);
        MOPHI_GPU_CALL(
            cudaMemcpyAsync(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
        // Full sync will make the time stamp the same
        m_device_mod_time = m_host_mod_time;
    }

    void ToDeviceAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Partial flavor aims for speed, no size check
        // When use partial send, users are responsible for managing the time stamp
        MOPHI_GPU_CALL(cudaMemcpyAsync(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
    }

    void ToHost() {
        assert(m_device_ptr && m_host_vec_ptr);
        MOPHI_GPU_CALL(cudaMemcpy(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost));
        // Full sync will make the time stamp the same
        m_host_mod_time = m_device_mod_time;
    }

    void ToHost(size_t start, size_t n) {
        assert(m_device_ptr && m_host_vec_ptr);
        // When use partial send, users are responsible for managing the time stamp
        MOPHI_GPU_CALL(
            cudaMemcpy(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void ToHostAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr && m_device_ptr);
        MOPHI_GPU_CALL(
            cudaMemcpyAsync(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        // Full sync will make the time stamp the same
        m_host_mod_time = m_device_mod_time;
    }

    void ToHostAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        // Async partial flavor aims for speed, no size check
        // When use partial send, users are responsible for managing the time stamp
        MOPHI_GPU_CALL(cudaMemcpyAsync(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
    }

    // Get a single value on host (if host data is stale, it will sync from device first)
    T GetVal(size_t start) {
        MakeReadyHost();
        return (*m_host_vec_ptr)[start];
    }

    // Get a portion of the vector on host (if host data is stale, it will sync from device first)
    std::vector<T> GetVal(size_t start, size_t n) {
        MakeReadyHost();
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    // Force sync a single value to host, then get it, without updating time stamps
    T GetVal_ForceSync(size_t start) {
        ToHost(start, 1);  // Partial ToHost to avoid updating time stamps
        return (*m_host_vec_ptr)[start];
    }

    // Force sync a portion of the vector to host, then get it, without updating time stamps
    std::vector<T> GetVal_ForceSync(size_t start, size_t n) {
        ToHost(start, n);  // Partial ToHost to avoid updating time stamps
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    // Set a single value on host, updating time stamps (will sync to device when used)
    void SetVal(const T& data, size_t start) {
        pre_host_access();
        (*m_host_vec_ptr)[start] = data;
    }

    // Set values using a vector on host, updating time stamps (will sync to device when used)
    void SetVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        pre_host_access();
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
    }

    // Set all values to a single value on host, updating time stamps (will sync to device when used)
    void SetVal(const T& data) {
        pre_host_access();
        std::fill(m_host_vec_ptr->begin(), m_host_vec_ptr->end(), data);
    }

    // Force set a single value on host and device, bypassing time stamp records
    void SetVal_ForceSync(const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        ToDevice(start, 1);  // Partial ToDevice to avoid updating time stamps
    }

    // Force set values using a vector on host and device, bypassing time stamp records
    void SetVal_ForceSync(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        ToDevice(start, count);  // Partial ToDevice to avoid updating time stamps
    }

    // Force set a single value on host and device, bypassing time stamp records, using async-ed stream
    void SetVal_ForceSync(cudaStream_t& stream, const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        ToDeviceAsync(stream, start, 1);  // Partial ToDeviceAsync to avoid updating time stamps
    }

    // Force set values using a vector on host and device, bypassing time stamp records, using async-ed stream
    void SetVal_ForceSync(cudaStream_t& stream, const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        ToDeviceAsync(stream, start, count);  // Partial ToDeviceAsync to avoid updating time stamps
    }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    // Get host or device size in bytes
    size_t GetNumBytes() const { return m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0; }

    T* host() {
        pre_host_access();
        return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr;
    }

    T* device() {
        pre_device_access();
        return m_device_ptr;
    }

    PinnedVector& GetHostVector() {
        pre_host_access();
        return *m_host_vec_ptr;
    }

    void BindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        update_bound_device_pointer();
    }

    void UnbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void AttachHostVector(const std::vector<T>& external_vec, bool deep_copy = true) {
        FreeHost();  // discard internal memory and update memory tracker
        if (deep_copy) {
            m_pinned_vec = std::make_unique<PinnedVector>(external_vec.begin(), external_vec.end());
            m_host_vec_ptr = m_pinned_vec.get();
            update_host_mem_counter(static_cast<ssize_t>(m_host_vec_ptr->size() * sizeof(T)));
        } else {
            m_host_vec_ptr = const_cast<PinnedVector*>(reinterpret_cast<const PinnedVector*>(&external_vec));
        }
        m_host_mod_time = ++m_access_time;  // Reset time stamps
    }

    // Manually declare that the vectors are in sync
    void DeclareSync() { m_host_mod_time = m_device_mod_time = ++m_access_time; }

    // Make sure device data is in sync, to be ready for kernels
    void MakeReadyDevice() {
        if (m_host_mod_time > m_device_mod_time) {
            ToDevice();
        }
    }

    // Make sure host data is in sync, to be ready for CPU access
    void MakeReadyHost() {
        if (m_device_mod_time > m_host_mod_time) {
            ToHost();
        }
    }

    T& operator[](size_t i) { return (host())[i]; }
    const T& operator[](size_t i) const { return (host())[i]; }

  private:
    std::unique_ptr<PinnedVector> m_pinned_vec = nullptr;
    PinnedVector* m_host_vec_ptr = nullptr;

    size_t m_host_mem_counter = 0;
    size_t m_device_mem_counter = 0;

    T* m_device_ptr = nullptr;
    size_t m_device_capacity = 0;

    T** m_bound_device_ptr = nullptr;

    uint64_t m_host_mod_time = 0;    // Tracks the last time host data was modified or accessed
    uint64_t m_device_mod_time = 0;  // Tracks the last time device data was modified or accessed
    uint64_t m_access_time = 0;      // Tracks the last time data was accessed or modified

    void ensure_host_vector(size_t n = 0) {
        // Guaranteed to be a non-meaningful update in the sense of values, no sync needed
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<PinnedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void pre_device_access() {
        MakeReadyDevice();
        // Assume a significant change will then be made to device data
        m_device_mod_time = ++m_access_time;
    }

    void pre_host_access() {
        MakeReadyHost();
        // Assume a significant change will then be made to host data
        m_host_mod_time = ++m_access_time;
    }

    void update_bound_device_pointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = m_device_ptr;
    }

    void update_host_mem_counter(ssize_t delta) { m_host_mem_counter += delta; }

    void update_device_mem_counter(ssize_t delta) { m_device_mem_counter += delta; }
};

// Pure device data type, usually used for scratching space
template <typename T>
class DeviceArray : private NonCopyable {
  public:
    using value_type = T;
    DeviceArray() {}

    explicit DeviceArray(size_t n) { resize(n); }

    ~DeviceArray() { free(); }

    // In practice, we use device array as temp arrays so we never really resize, let alone preserving existing data
    void resize(size_t n, bool allow_shrink = false) {
        if (!allow_shrink && m_capacity >= n)
            return;
        T* new_device_ptr = nullptr;
        device_ptr_alloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount
        if (m_data && m_capacity > 0) {
            size_t copy_count = std::min(n, m_capacity);
            MOPHI_GPU_CALL(cudaMemcpy(new_device_ptr, m_data, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        // Free old memory and update bookkeeping
        update_mem_counter(-(ssize_t)(m_capacity * sizeof(T)));
        device_ptr_dealloc(m_data);

        m_data = new_device_ptr;

        update_mem_counter(static_cast<ssize_t>(n * sizeof(T)));
        m_capacity = n;
    }

    void free() {
        device_ptr_dealloc(m_data);
        update_mem_counter(-(ssize_t)(m_capacity * sizeof(T)));
        m_data = nullptr;
        m_capacity = 0;
    }

    size_t size() const { return m_capacity; }

    // Get host or device size in bytes
    size_t GetNumBytes() const { return m_capacity * sizeof(T); }

    T* data() { return m_data; }

    const T* data() const { return m_data; }

    // Force set a single value on device
    void SetVal(const T& data, size_t idx) {
        MOPHI_GPU_CALL(cudaMemcpy(m_data + idx, &data, sizeof(T), cudaMemcpyHostToDevice));
    }

    // Force set values using a vector on device
    void SetVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        MOPHI_GPU_CALL(cudaMemcpy(m_data + start, data.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Force set all values to a single value on device
    void SetVal(const T& data) {
        // This method is not super efficient, but it's not meant for performance-critical paths
        std::vector<T> temp(size(), data);
        MOPHI_GPU_CALL(cudaMemcpy(m_data, temp.data(), size() * sizeof(T), cudaMemcpyHostToDevice));
    }

  private:
    T* m_data = nullptr;
    size_t m_capacity = 0;
    size_t m_mem_counter = 0;

    void update_mem_counter(ssize_t delta) { m_mem_counter += delta; }
};

}  // namespace mophi

#endif
