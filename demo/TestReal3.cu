//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cmath>
#include <cassert>
#include "cuda_runtime.h"
#include <core/Real3.hpp>
#include <core/Logger.hpp>

using namespace mophi;

__global__ void test_device_ops(Real3f* out) {
    Real3f a(1.0f, 2.0f, 3.0f);
    Real3f b(2.0f, 2.0f, 2.0f);
    Real3f c = (a + b) * 0.5f;

    out[0] = c;  // (3,4,5) * 0.5 = (1.5, 2.0, 2.5)
}

void test_device() {
    Real3f* d_out;
    Real3f h_out;
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_out, sizeof(Real3f)));
    test_device_ops<<<1, 1>>>(d_out);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
    MOPHI_GPU_CALL(cudaMemcpy(&h_out, d_out, sizeof(Real3f), cudaMemcpyDeviceToHost));

    assert(std::abs(h_out.x() - 1.5f) < 1e-5);
    assert(std::abs(h_out.y() - 2.0f) < 1e-5);
    assert(std::abs(h_out.z() - 2.5f) < 1e-5);

    MOPHI_GPU_CALL(cudaFree(d_out));
    std::cout << "Device-side Real3f test passed.\n";
}

void test_host_ops() {
    std::cout << "Real3f's size: " << sizeof(Real3f) << " bytes\n";
    std::cout << "Real3d's size: " << sizeof(Real3d) << " bytes\n";

    Real3f a(1.0f, 2.0f, 3.0f);
    Real3f b(4.0f, 5.0f, 6.0f);

    // Basic arithmetic
    Real3f c = a + b;
    assert(c.x() == 5.0f && c.y() == 7.0f && c.z() == 9.0f);

    Real3f d = b - a;
    assert(d.x() == 3.0f && d.y() == 3.0f && d.z() == 3.0f);

    Real3f e = a * 2.0f;
    assert(e.x() == 2.0f && e.y() == 4.0f && e.z() == 6.0f);

    Real3f f = b / 2.0f;
    assert(f.x() == 2.0f && f.y() == 2.5f && f.z() == 3.0f);

    // Compound assignment
    a += b;
    assert(a.x() == 5.0f && a.y() == 7.0f && a.z() == 9.0f);

    b -= Real3f(1.0f, 1.0f, 1.0f);
    assert(b.x() == 3.0f && b.y() == 4.0f && b.z() == 5.0f);

    // Unary minus
    Real3f g = -a;
    assert(g.x() == -5.0f && g.y() == -7.0f && g.z() == -9.0f);

    // Equality
    assert(Real3f(1, 2, 3) == Real3f(1, 2, 3));
    assert(Real3f(1, 2, 3) != Real3f(3, 2, 1));

    // Dot product and norm (if defined)
    float dot = Vdot(Real3f(1, 0, 0), Real3f(0, 1, 0));
    assert(dot == 0.0f);

    float norm_sq = Real3f(1, 2, 2).Length2();  // should be 9
    assert(norm_sq == 9.0f);

    float norm = Real3f(3, 4, 0).Length();  // should be 5
    assert(std::abs(norm - 5.0f) < 1e-5);

    std::cout << "All host-side Real3f tests passed.\n";
}

int main() {
    test_host_ops();
    test_device();
    return 0;
}
