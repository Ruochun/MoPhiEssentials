//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_STATIC_DEVICE_UTIL_CUH
#define MOPHI_STATIC_DEVICE_UTIL_CUH

#include <common/Defines.hpp>

namespace mophi {

// ========================================================================
// Some simple, static device-side utilities are here, and they need a place
// to live with cuda compilation environment
// ========================================================================

template <typename T1, typename T2, typename T3>
__global__ void one_num_add(T1* res, const T2* a, const T3* b) {
    T1 T1a = (T1)(*a);
    T1 T1b = (T1)(*b);
    *res = T1a + T1b;
}
template <typename T1, typename T2, typename T3>
void device_add(T1* res, const T2* a, const T3* b, cudaStream_t& this_stream) {
    one_num_add<T1, T2, T3><<<1, 1, 0, this_stream>>>(res, a, b);
    MOPHI_GPU_CALL(cudaStreamSynchronize(this_stream));
}

template <typename T1, typename T2>
__global__ void one_num_assign(T1* res, const T2* a) {
    T1 T1a = (T1)(*a);
    *res = T1a;
}
template <typename T1, typename T2>
void device_assign(T1* res, const T2* a, cudaStream_t& this_stream) {
    one_num_assign<T1, T2><<<1, 1, 0, this_stream>>>(res, a);
    MOPHI_GPU_CALL(cudaStreamSynchronize(this_stream));
}

}  // namespace mophi

#endif
