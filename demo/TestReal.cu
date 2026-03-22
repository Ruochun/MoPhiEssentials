//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cmath>
#include <cassert>
#include "cuda_runtime.h"
#include "core/Real3.hpp"
#include "core/Real4.hpp"
#include "core/Logger.hpp"

using namespace mophi;

// =============================================================================
// Real3 tests
// =============================================================================

__global__ void test_real3_device_ops(Real3f* out) {
    Real3f a(1.0f, 2.0f, 3.0f);
    Real3f b(2.0f, 2.0f, 2.0f);
    Real3f c = (a + b) * 0.5f;

    out[0] = c;  // (3,4,5) * 0.5 = (1.5, 2.0, 2.5)
}

void test_real3_device() {
    Real3f* d_out;
    Real3f h_out;
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_out, sizeof(Real3f)));
    test_real3_device_ops<<<1, 1>>>(d_out);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
    MOPHI_GPU_CALL(cudaMemcpy(&h_out, d_out, sizeof(Real3f), cudaMemcpyDeviceToHost));

    assert(std::abs(h_out.x() - 1.5f) < 1e-5);
    assert(std::abs(h_out.y() - 2.0f) < 1e-5);
    assert(std::abs(h_out.z() - 2.5f) < 1e-5);

    MOPHI_GPU_CALL(cudaFree(d_out));
    std::cout << "Device-side Real3f test passed.\n";
}

void test_real3_host_ops() {
    std::cout << "Real3f size: " << sizeof(Real3f) << " bytes\n";
    std::cout << "Real3d size: " << sizeof(Real3d) << " bytes\n";

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

    // Dot product and norms
    float dot = Vdot(Real3f(1, 0, 0), Real3f(0, 1, 0));
    assert(dot == 0.0f);

    float norm_sq = Real3f(1, 2, 2).Length2();
    assert(norm_sq == 9.0f);

    float norm = Real3f(3, 4, 0).Length();
    assert(std::abs(norm - 5.0f) < 1e-5);

    // Utility methods
    Real3f v(1.5f, -2.7f, 3.2f);

    Real3f abs_v = v.Abs();
    assert(abs_v.x() > 0.0f && abs_v.y() > 0.0f && abs_v.z() > 0.0f);

    Real3f fl = v.Floor();
    assert(fl.x() == 1.0f && fl.y() == -3.0f && fl.z() == 3.0f);

    Real3f fr = v.Frac();
    assert(std::abs(fr.x() - 0.5f) < 1e-5f);

    Real3f cl = v.Clamp(-1.0f, 2.0f);
    assert(cl.x() == 1.5f && cl.y() == -1.0f && cl.z() == 2.0f);

    Real3f lerped = Real3f(0, 0, 0).Lerp(Real3f(2, 4, 6), 0.5f);
    assert(std::abs(lerped.x() - 1.0f) < 1e-5f);
    assert(std::abs(lerped.y() - 2.0f) < 1e-5f);

    Real3f inc(1.0f, -1.0f, 0.0f);
    Real3f n(0.0f, 1.0f, 0.0f);
    Real3f reflected = inc.Reflect(n);
    assert(std::abs(reflected.x() - 1.0f) < 1e-5f);
    assert(std::abs(reflected.y() - 1.0f) < 1e-5f);
    assert(std::abs(reflected.z()) < 1e-5f);

    Real3f fmod_v = Real3f(5.5f, -3.2f, 7.0f).Fmod(Real3f(3.0f, 2.0f, 4.0f));
    assert(std::abs(fmod_v.x() - std::fmod(5.5f, 3.0f)) < 1e-5f);

    std::cout << "All host-side Real3 tests passed.\n";
}

// =============================================================================
// Real4 tests
// =============================================================================

__global__ void test_real4_device_ops(Real4f* out) {
    Real4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Real4f b(2.0f, 2.0f, 2.0f, 2.0f);
    Real4f c = (a + b) * 0.5f;

    out[0] = c;  // (3,4,5,6) * 0.5 = (1.5, 2.0, 2.5, 3.0)
}

void test_real4_device() {
    Real4f* d_out;
    Real4f h_out;
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_out, sizeof(Real4f)));
    test_real4_device_ops<<<1, 1>>>(d_out);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
    MOPHI_GPU_CALL(cudaMemcpy(&h_out, d_out, sizeof(Real4f), cudaMemcpyDeviceToHost));

    assert(std::abs(h_out.x() - 1.5f) < 1e-5);
    assert(std::abs(h_out.y() - 2.0f) < 1e-5);
    assert(std::abs(h_out.z() - 2.5f) < 1e-5);
    assert(std::abs(h_out.w() - 3.0f) < 1e-5);

    MOPHI_GPU_CALL(cudaFree(d_out));
    std::cout << "Device-side Real4f test passed.\n";
}

void test_real4_host_ops() {
    std::cout << "Real4f size: " << sizeof(Real4f) << " bytes\n";
    std::cout << "Real4d size: " << sizeof(Real4d) << " bytes\n";

    Real4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Real4f b(4.0f, 3.0f, 2.0f, 1.0f);

    // Basic arithmetic
    Real4f sum = a + b;
    assert(sum.x() == 5.0f && sum.y() == 5.0f && sum.z() == 5.0f && sum.w() == 5.0f);

    Real4f diff = b - a;
    assert(diff.x() == 3.0f && diff.y() == 1.0f && diff.z() == -1.0f && diff.w() == -3.0f);

    Real4f scaled = a * 2.0f;
    assert(scaled.x() == 2.0f && scaled.y() == 4.0f && scaled.z() == 6.0f && scaled.w() == 8.0f);

    Real4f divided = a / 2.0f;
    assert(std::abs(divided.x() - 0.5f) < 1e-5f && std::abs(divided.w() - 2.0f) < 1e-5f);

    // Compound assignment
    Real4f tmp(1, 2, 3, 4);
    tmp += Real4f(1, 1, 1, 1);
    assert(tmp.x() == 2.0f && tmp.w() == 5.0f);

    tmp -= Real4f(1, 1, 1, 1);
    assert(tmp.x() == 1.0f && tmp.w() == 4.0f);

    // Unary minus
    Real4f neg = -a;
    assert(neg.x() == -1.0f && neg.w() == -4.0f);

    // Equality
    assert(Real4f(1, 2, 3, 4) == Real4f(1, 2, 3, 4));
    assert(Real4f(1, 2, 3, 4) != Real4f(4, 3, 2, 1));

    // Subscript
    assert(a[0] == 1.0f && a[3] == 4.0f);

    // Dot product
    float dot = Real4f(1, 0, 0, 0) ^ Real4f(0, 1, 0, 0);
    assert(dot == 0.0f);

    float dot2 = Real4f(1, 2, 3, 4).Dot(Real4f(1, 2, 3, 4));
    assert(std::abs(dot2 - 30.0f) < 1e-5f);  // 1+4+9+16 = 30

    // Norms
    float len2 = Real4f(1, 2, 2, 0).Length2();
    assert(std::abs(len2 - 9.0f) < 1e-5f);

    float len = Real4f(0, 0, 3, 4).Length();
    assert(std::abs(len - 5.0f) < 1e-5f);

    // Normalize
    Real4f nn = Real4f(0, 0, 3, 4).GetNormalized();
    assert(std::abs(nn.Length() - 1.0f) < 1e-5f);

    // Utility methods
    Real4f v(1.5f, -2.7f, 3.2f, -0.4f);

    Real4f abs_v = v.Abs();
    assert(abs_v.x() > 0.0f && abs_v.y() > 0.0f && abs_v.z() > 0.0f && abs_v.w() > 0.0f);

    Real4f fl = v.Floor();
    assert(fl.x() == 1.0f && fl.y() == -3.0f && fl.z() == 3.0f && fl.w() == -1.0f);

    Real4f fr = v.Frac();
    assert(std::abs(fr.x() - 0.5f) < 1e-5f);

    Real4f cl = v.Clamp(-1.0f, 2.0f);
    assert(cl.x() == 1.5f && cl.y() == -1.0f && cl.z() == 2.0f && cl.w() == -0.4f);

    Real4f cl2 = v.Clamp(Real4f(0, -2, 0, -0.5f), Real4f(2, 2, 2.5f, 0));
    assert(cl2.x() == 1.5f && cl2.y() == -2.0f && cl2.z() == 2.0f);
    assert(std::abs(cl2.w() - (-0.4f)) < 1e-5f);

    Real4f lerped = Real4f(0, 0, 0, 0).Lerp(Real4f(2, 4, 6, 8), 0.5f);
    assert(std::abs(lerped.x() - 1.0f) < 1e-5f);
    assert(std::abs(lerped.w() - 4.0f) < 1e-5f);

    Real4f fmod_v = Real4f(5.5f, -3.2f, 7.0f, 9.1f).Fmod(Real4f(3.0f, 2.0f, 4.0f, 5.0f));
    assert(std::abs(fmod_v.x() - std::fmod(5.5f, 3.0f)) < 1e-5f);
    assert(std::abs(fmod_v.w() - std::fmod(9.1f, 5.0f)) < 1e-4f);

    std::cout << "All host-side Real4 tests passed.\n";
}

// =============================================================================

int main() {
    test_real3_host_ops();
    test_real3_device();
    test_real4_host_ops();
    test_real4_device();
    std::cout << "All TestReal tests passed.\n";
    return 0;
}
