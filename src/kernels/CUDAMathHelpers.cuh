/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef MOPHI_HELPER_MATH_H
#define MOPHI_HELPER_MATH_H

#if defined(_WIN32) || defined(_WIN64)
    #undef max
    #undef min
    #undef strtok_r
#endif

#include "cuda_runtime.h"

#ifndef EXIT_WAIVED
    #define EXIT_WAIVED 2
#endif

#ifndef MOPHI_HD
    #ifdef __CUDACC__
        #define MOPHI_HD __host__ __device__
    #else
        #define MOPHI_HD
    #endif
#endif

using uint = unsigned int;
using ushort = unsigned short;

#ifndef __CUDACC__
    ////////////////////////////////////////////////////////////////////////////////
    // override implementations of CUDA functions
    ////////////////////////////////////////////////////////////////////////////////
    #include <cmath>
using std::fminf;
using std::fmaxf;
using std::max;
using std::min;

inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 make_float2(float s) {
    return make_float2(s, s);
}
inline MOPHI_HD float2 make_float2(float3 a) {
    return make_float2(a.x, a.y);
}
inline MOPHI_HD float2 make_float2(int2 a) {
    return make_float2(float(a.x), float(a.y));
}
inline MOPHI_HD float2 make_float2(uint2 a) {
    return make_float2(float(a.x), float(a.y));
}

inline MOPHI_HD int2 make_int2(int s) {
    return make_int2(s, s);
}
inline MOPHI_HD int2 make_int2(int3 a) {
    return make_int2(a.x, a.y);
}
inline MOPHI_HD int2 make_int2(uint2 a) {
    return make_int2(int(a.x), int(a.y));
}
inline MOPHI_HD int2 make_int2(float2 a) {
    return make_int2(int(a.x), int(a.y));
}

inline MOPHI_HD uint2 make_uint2(uint s) {
    return make_uint2(s, s);
}
inline MOPHI_HD uint2 make_uint2(uint3 a) {
    return make_uint2(a.x, a.y);
}
inline MOPHI_HD uint2 make_uint2(int2 a) {
    return make_uint2(uint(a.x), uint(a.y));
}

inline MOPHI_HD float3 make_float3(float s) {
    return make_float3(s, s, s);
}
inline MOPHI_HD float3 make_float3(float2 a) {
    return make_float3(a.x, a.y, 0.0f);
}
inline MOPHI_HD float3 make_float3(float2 a, float s) {
    return make_float3(a.x, a.y, s);
}
inline MOPHI_HD float3 make_float3(float4 a) {
    return make_float3(a.x, a.y, a.z);
}
inline MOPHI_HD float3 make_float3(int3 a) {
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline MOPHI_HD float3 make_float3(uint3 a) {
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline MOPHI_HD int3 make_int3(int s) {
    return make_int3(s, s, s);
}
inline MOPHI_HD int3 make_int3(int2 a) {
    return make_int3(a.x, a.y, 0);
}
inline MOPHI_HD int3 make_int3(int2 a, int s) {
    return make_int3(a.x, a.y, s);
}
inline MOPHI_HD int3 make_int3(uint3 a) {
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline MOPHI_HD int3 make_int3(float3 a) {
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline MOPHI_HD uint3 make_uint3(uint s) {
    return make_uint3(s, s, s);
}
inline MOPHI_HD uint3 make_uint3(uint2 a) {
    return make_uint3(a.x, a.y, 0);
}
inline MOPHI_HD uint3 make_uint3(uint2 a, uint s) {
    return make_uint3(a.x, a.y, s);
}
inline MOPHI_HD uint3 make_uint3(uint4 a) {
    return make_uint3(a.x, a.y, a.z);
}
inline MOPHI_HD uint3 make_uint3(int3 a) {
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline MOPHI_HD float4 make_float4(float s) {
    return make_float4(s, s, s, s);
}
inline MOPHI_HD float4 make_float4(float3 a) {
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline MOPHI_HD float4 make_float4(float3 a, float w) {
    return make_float4(a.x, a.y, a.z, w);
}
inline MOPHI_HD float4 make_float4(int4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline MOPHI_HD float4 make_float4(uint4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline MOPHI_HD int4 make_int4(int s) {
    return make_int4(s, s, s, s);
}
inline MOPHI_HD int4 make_int4(int3 a) {
    return make_int4(a.x, a.y, a.z, 0);
}
inline MOPHI_HD int4 make_int4(int3 a, int w) {
    return make_int4(a.x, a.y, a.z, w);
}
inline MOPHI_HD int4 make_int4(uint4 a) {
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline MOPHI_HD int4 make_int4(float4 a) {
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

inline MOPHI_HD uint4 make_uint4(uint s) {
    return make_uint4(s, s, s, s);
}
inline MOPHI_HD uint4 make_uint4(uint3 a) {
    return make_uint4(a.x, a.y, a.z, 0);
}
inline MOPHI_HD uint4 make_uint4(uint3 a, uint w) {
    return make_uint4(a.x, a.y, a.z, w);
}
inline MOPHI_HD uint4 make_uint4(int4 a) {
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 operator-(float2& a) {
    return make_float2(-a.x, -a.y);
}
inline MOPHI_HD int2 operator-(int2& a) {
    return make_int2(-a.x, -a.y);
}
inline MOPHI_HD float3 operator-(float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
inline MOPHI_HD int3 operator-(int3& a) {
    return make_int3(-a.x, -a.y, -a.z);
}
inline MOPHI_HD float4 operator-(float4& a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline MOPHI_HD int4 operator-(int4& a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
inline MOPHI_HD void operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}
inline MOPHI_HD float2 operator+(float2 a, float b) {
    return make_float2(a.x + b, a.y + b);
}
inline MOPHI_HD float2 operator+(float b, float2 a) {
    return make_float2(a.x + b, a.y + b);
}
inline MOPHI_HD void operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
}

inline MOPHI_HD int2 operator+(int2 a, int2 b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
inline MOPHI_HD void operator+=(int2& a, int2 b) {
    a.x += b.x;
    a.y += b.y;
}
inline MOPHI_HD int2 operator+(int2 a, int b) {
    return make_int2(a.x + b, a.y + b);
}
inline MOPHI_HD int2 operator+(int b, int2 a) {
    return make_int2(a.x + b, a.y + b);
}
inline MOPHI_HD void operator+=(int2& a, int b) {
    a.x += b;
    a.y += b;
}

inline MOPHI_HD uint2 operator+(uint2 a, uint2 b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline MOPHI_HD void operator+=(uint2& a, uint2 b) {
    a.x += b.x;
    a.y += b.y;
}
inline MOPHI_HD uint2 operator+(uint2 a, uint b) {
    return make_uint2(a.x + b, a.y + b);
}
inline MOPHI_HD uint2 operator+(uint b, uint2 a) {
    return make_uint2(a.x + b, a.y + b);
}
inline MOPHI_HD void operator+=(uint2& a, uint b) {
    a.x += b;
    a.y += b;
}

inline MOPHI_HD float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline MOPHI_HD void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline MOPHI_HD void operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline MOPHI_HD int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline MOPHI_HD void operator+=(int3& a, int3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD int3 operator+(int3 a, int b) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline MOPHI_HD void operator+=(int3& a, int b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline MOPHI_HD uint3 operator+(uint3 a, uint3 b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline MOPHI_HD void operator+=(uint3& a, uint3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD uint3 operator+(uint3 a, uint b) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline MOPHI_HD void operator+=(uint3& a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline MOPHI_HD int3 operator+(int b, int3 a) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline MOPHI_HD uint3 operator+(uint b, uint3 a) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline MOPHI_HD float3 operator+(float b, float3 a) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline MOPHI_HD float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline MOPHI_HD void operator+=(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline MOPHI_HD float4 operator+(float4 a, float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD void operator+=(float4& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline MOPHI_HD int4 operator+(int4 a, int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline MOPHI_HD void operator+=(int4& a, int4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline MOPHI_HD int4 operator+(int4 a, int b) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD int4 operator+(int b, int4 a) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD void operator+=(int4& a, int b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline MOPHI_HD uint4 operator+(uint4 a, uint4 b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline MOPHI_HD void operator+=(uint4& a, uint4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline MOPHI_HD uint4 operator+(uint4 a, uint b) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD uint4 operator+(uint b, uint4 a) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline MOPHI_HD void operator+=(uint4& a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
inline MOPHI_HD void operator-=(float2& a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
inline MOPHI_HD float2 operator-(float2 a, float b) {
    return make_float2(a.x - b, a.y - b);
}
inline MOPHI_HD float2 operator-(float b, float2 a) {
    return make_float2(b - a.x, b - a.y);
}
inline MOPHI_HD void operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
}

inline MOPHI_HD int2 operator-(int2 a, int2 b) {
    return make_int2(a.x - b.x, a.y - b.y);
}
inline MOPHI_HD void operator-=(int2& a, int2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
inline MOPHI_HD int2 operator-(int2 a, int b) {
    return make_int2(a.x - b, a.y - b);
}
inline MOPHI_HD int2 operator-(int b, int2 a) {
    return make_int2(b - a.x, b - a.y);
}
inline MOPHI_HD void operator-=(int2& a, int b) {
    a.x -= b;
    a.y -= b;
}

inline MOPHI_HD uint2 operator-(uint2 a, uint2 b) {
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline MOPHI_HD void operator-=(uint2& a, uint2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
inline MOPHI_HD uint2 operator-(uint2 a, uint b) {
    return make_uint2(a.x - b, a.y - b);
}
inline MOPHI_HD uint2 operator-(uint b, uint2 a) {
    return make_uint2(b - a.x, b - a.y);
}
inline MOPHI_HD void operator-=(uint2& a, uint b) {
    a.x -= b;
    a.y -= b;
}

inline MOPHI_HD float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline MOPHI_HD void operator-=(float3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline MOPHI_HD float3 operator-(float b, float3 a) {
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline MOPHI_HD void operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline MOPHI_HD int3 operator-(int3 a, int3 b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline MOPHI_HD void operator-=(int3& a, int3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD int3 operator-(int3 a, int b) {
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline MOPHI_HD int3 operator-(int b, int3 a) {
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline MOPHI_HD void operator-=(int3& a, int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline MOPHI_HD uint3 operator-(uint3 a, uint3 b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline MOPHI_HD void operator-=(uint3& a, uint3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD uint3 operator-(uint3 a, uint b) {
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline MOPHI_HD uint3 operator-(uint b, uint3 a) {
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline MOPHI_HD void operator-=(uint3& a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline MOPHI_HD float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline MOPHI_HD void operator-=(float4& a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline MOPHI_HD float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline MOPHI_HD void operator-=(float4& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline MOPHI_HD int4 operator-(int4 a, int4 b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline MOPHI_HD void operator-=(int4& a, int4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline MOPHI_HD int4 operator-(int4 a, int b) {
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline MOPHI_HD int4 operator-(int b, int4 a) {
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline MOPHI_HD void operator-=(int4& a, int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline MOPHI_HD uint4 operator-(uint4 a, uint4 b) {
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline MOPHI_HD void operator-=(uint4& a, uint4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline MOPHI_HD uint4 operator-(uint4 a, uint b) {
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline MOPHI_HD uint4 operator-(uint b, uint4 a) {
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline MOPHI_HD void operator-=(uint4& a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
inline MOPHI_HD void operator*=(float2& a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
inline MOPHI_HD float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}
inline MOPHI_HD float2 operator*(float b, float2 a) {
    return make_float2(b * a.x, b * a.y);
}
inline MOPHI_HD void operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
}

inline MOPHI_HD int2 operator*(int2 a, int2 b) {
    return make_int2(a.x * b.x, a.y * b.y);
}
inline MOPHI_HD void operator*=(int2& a, int2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
inline MOPHI_HD int2 operator*(int2 a, int b) {
    return make_int2(a.x * b, a.y * b);
}
inline MOPHI_HD int2 operator*(int b, int2 a) {
    return make_int2(b * a.x, b * a.y);
}
inline MOPHI_HD void operator*=(int2& a, int b) {
    a.x *= b;
    a.y *= b;
}

inline MOPHI_HD uint2 operator*(uint2 a, uint2 b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline MOPHI_HD void operator*=(uint2& a, uint2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
inline MOPHI_HD uint2 operator*(uint2 a, uint b) {
    return make_uint2(a.x * b, a.y * b);
}
inline MOPHI_HD uint2 operator*(uint b, uint2 a) {
    return make_uint2(b * a.x, b * a.y);
}
inline MOPHI_HD void operator*=(uint2& a, uint b) {
    a.x *= b;
    a.y *= b;
}

inline MOPHI_HD float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline MOPHI_HD void operator*=(float3& a, float3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline MOPHI_HD float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline MOPHI_HD float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline MOPHI_HD void operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline MOPHI_HD int3 operator*(int3 a, int3 b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline MOPHI_HD void operator*=(int3& a, int3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline MOPHI_HD int3 operator*(int3 a, int b) {
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline MOPHI_HD int3 operator*(int b, int3 a) {
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline MOPHI_HD void operator*=(int3& a, int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline MOPHI_HD uint3 operator*(uint3 a, uint3 b) {
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline MOPHI_HD void operator*=(uint3& a, uint3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline MOPHI_HD uint3 operator*(uint3 a, uint b) {
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline MOPHI_HD uint3 operator*(uint b, uint3 a) {
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline MOPHI_HD void operator*=(uint3& a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline MOPHI_HD float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline MOPHI_HD void operator*=(float4& a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline MOPHI_HD float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline MOPHI_HD float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline MOPHI_HD void operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline MOPHI_HD int4 operator*(int4 a, int4 b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline MOPHI_HD void operator*=(int4& a, int4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline MOPHI_HD int4 operator*(int4 a, int b) {
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline MOPHI_HD int4 operator*(int b, int4 a) {
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline MOPHI_HD void operator*=(int4& a, int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline MOPHI_HD uint4 operator*(uint4 a, uint4 b) {
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline MOPHI_HD void operator*=(uint4& a, uint4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline MOPHI_HD uint4 operator*(uint4 a, uint b) {
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline MOPHI_HD uint4 operator*(uint b, uint4 a) {
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline MOPHI_HD void operator*=(uint4& a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}
inline MOPHI_HD void operator/=(float2& a, float2 b) {
    a.x /= b.x;
    a.y /= b.y;
}
inline MOPHI_HD float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}
inline MOPHI_HD void operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
}
inline MOPHI_HD float2 operator/(float b, float2 a) {
    return make_float2(b / a.x, b / a.y);
}

inline MOPHI_HD float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline MOPHI_HD void operator/=(float3& a, float3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline MOPHI_HD float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline MOPHI_HD void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline MOPHI_HD float3 operator/(float b, float3 a) {
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline MOPHI_HD float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline MOPHI_HD void operator/=(float4& a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline MOPHI_HD float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline MOPHI_HD void operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline MOPHI_HD float4 operator/(float b, float4 a) {
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 fminf(float2 a, float2 b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
inline MOPHI_HD float3 fminf(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline MOPHI_HD float4 fminf(float4 a, float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline MOPHI_HD int2 min(int2 a, int2 b) {
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}
inline MOPHI_HD int3 min(int3 a, int3 b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline MOPHI_HD int4 min(int4 a, int4 b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline MOPHI_HD uint2 min(uint2 a, uint2 b) {
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
inline MOPHI_HD uint3 min(uint3 a, uint3 b) {
    return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline MOPHI_HD uint4 min(uint4 a, uint4 b) {
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 fmaxf(float2 a, float2 b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
inline MOPHI_HD float3 fmaxf(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline MOPHI_HD float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline MOPHI_HD int2 max(int2 a, int2 b) {
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}
inline MOPHI_HD int3 max(int3 a, int3 b) {
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline MOPHI_HD int4 max(int4 a, int4 b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline MOPHI_HD uint2 max(uint2 a, uint2 b) {
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
inline MOPHI_HD uint3 max(uint3 a, uint3 b) {
    return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline MOPHI_HD uint4 max(uint4 a, uint4 b) {
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// logical
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD bool float_near(const float& a, const float& b, float tol = 1e-6f) {
    return fabs(a - b) < tol;
}
inline MOPHI_HD bool float3_near(const float3& a, const float3& b, float tol = 1e-6f) {
    return float_near(a.x, b.x, tol) && float_near(a.y, b.y, tol) && float_near(a.z, b.z, tol);
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float lerp(float a, float b, float t) {
    return a + t * (b - a);
}
inline MOPHI_HD float2 lerp(float2 a, float2 b, float t) {
    return a + t * (b - a);
}
inline MOPHI_HD float3 lerp(float3 a, float3 b, float t) {
    return a + t * (b - a);
}
inline MOPHI_HD float4 lerp(float4 a, float4 b, float t) {
    return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float clamp(float f, float a, float b) {
    return fmaxf(a, fminf(f, b));
}
inline MOPHI_HD int clamp(int f, int a, int b) {
    return max(a, min(f, b));
}
inline MOPHI_HD uint clamp(uint f, uint a, uint b) {
    return max(a, min(f, b));
}

inline MOPHI_HD float2 clamp(float2 v, float a, float b) {
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline MOPHI_HD float2 clamp(float2 v, float2 a, float2 b) {
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline MOPHI_HD float3 clamp(float3 v, float a, float b) {
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline MOPHI_HD float3 clamp(float3 v, float3 a, float3 b) {
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline MOPHI_HD float4 clamp(float4 v, float a, float b) {
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline MOPHI_HD float4 clamp(float4 v, float4 a, float4 b) {
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline MOPHI_HD int2 clamp(int2 v, int a, int b) {
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline MOPHI_HD int2 clamp(int2 v, int2 a, int2 b) {
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline MOPHI_HD int3 clamp(int3 v, int a, int b) {
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline MOPHI_HD int3 clamp(int3 v, int3 a, int3 b) {
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline MOPHI_HD int4 clamp(int4 v, int a, int b) {
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline MOPHI_HD int4 clamp(int4 v, int4 a, int4 b) {
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline MOPHI_HD uint2 clamp(uint2 v, uint a, uint b) {
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline MOPHI_HD uint2 clamp(uint2 v, uint2 a, uint2 b) {
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline MOPHI_HD uint3 clamp(uint3 v, uint a, uint b) {
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline MOPHI_HD uint3 clamp(uint3 v, uint3 a, uint3 b) {
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline MOPHI_HD uint4 clamp(uint4 v, uint a, uint b) {
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline MOPHI_HD uint4 clamp(uint4 v, uint4 a, uint4 b) {
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}
inline MOPHI_HD float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline MOPHI_HD float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline MOPHI_HD int dot(int2 a, int2 b) {
    return a.x * b.x + a.y * b.y;
}
inline MOPHI_HD int dot(int3 a, int3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline MOPHI_HD int dot(int4 a, int4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline MOPHI_HD uint dot(uint2 a, uint2 b) {
    return a.x * b.x + a.y * b.y;
}
inline MOPHI_HD uint dot(uint3 a, uint3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline MOPHI_HD uint dot(uint4 a, uint4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float length(float2 v) {
    return sqrtf(dot(v, v));
}
inline MOPHI_HD float length(float3 v) {
    return sqrtf(dot(v, v));
}
inline MOPHI_HD float length(float4 v) {
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 normalize(float2 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline MOPHI_HD float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline MOPHI_HD float4 normalize(float4 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 floorf(float2 v) {
    return make_float2(floorf(v.x), floorf(v.y));
}
inline MOPHI_HD float3 floorf(float3 v) {
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline MOPHI_HD float4 floorf(float4 v) {
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float fracf(float v) {
    return v - floorf(v);
}
inline MOPHI_HD float2 fracf(float2 v) {
    return make_float2(fracf(v.x), fracf(v.y));
}
inline MOPHI_HD float3 fracf(float3 v) {
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline MOPHI_HD float4 fracf(float4 v) {
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 fmodf(float2 a, float2 b) {
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline MOPHI_HD float3 fmodf(float3 a, float3 b) {
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline MOPHI_HD float4 fmodf(float4 a, float4 b) {
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float2 fabs(float2 v) {
    return make_float2(fabs(v.x), fabs(v.y));
}
inline MOPHI_HD float3 fabs(float3 v) {
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline MOPHI_HD float4 fabs(float4 v) {
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline MOPHI_HD int2 abs(int2 v) {
    return make_int2(abs(v.x), abs(v.y));
}
inline MOPHI_HD int3 abs(int3 v) {
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline MOPHI_HD int4 abs(int4 v) {
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float3 reflect(float3 i, float3 n) {
    return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD float smoothstep(float a, float b, float x) {
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (3.0f - (2.0f * y)));
}
inline MOPHI_HD float2 smoothstep(float2 a, float2 b, float2 x) {
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}
inline MOPHI_HD float3 smoothstep(float3 a, float3 b, float3 x) {
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}
inline MOPHI_HD float4 smoothstep(float4 a, float4 b, float4 x) {
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}

////////////////////////////////////////////////////////////////////////////////
// A few float3 and double3 operators are not in the cuda toolkit are added by Ruochun
////////////////////////////////////////////////////////////////////////////////

inline MOPHI_HD double3 cross(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline MOPHI_HD double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline MOPHI_HD float dot(double3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline MOPHI_HD double dot(double4 a, double4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline MOPHI_HD float dot(double4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline MOPHI_HD double length(double3 v) {
    return sqrt(dot(v, v));
}
inline MOPHI_HD double length(double4 v) {
    return sqrt(dot(v, v));
}

// Addition and subtraction

inline MOPHI_HD double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline MOPHI_HD double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline MOPHI_HD float3 operator+(double3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline MOPHI_HD float3 operator-(double3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline MOPHI_HD void operator+=(double3& a, double3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD void operator-=(double3& a, double3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD void operator+=(double3& a, double b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline MOPHI_HD void operator-=(double3& a, double b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline MOPHI_HD void operator+=(float3& a, double3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD void operator-=(float3& a, double3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD void operator+=(float3& a, double b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline MOPHI_HD void operator-=(float3& a, double b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline MOPHI_HD void operator+=(double3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline MOPHI_HD void operator-=(double3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline MOPHI_HD void operator+=(double3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline MOPHI_HD void operator-=(double3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

// Multiplication

inline MOPHI_HD double3 operator*(double3 a, double3 b) {
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline MOPHI_HD void operator*=(double3& a, double3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline MOPHI_HD double3 operator*(double3 a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline MOPHI_HD double3 operator*(double b, double3 a) {
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline MOPHI_HD void operator*=(double3& a, double b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

// Division

inline MOPHI_HD float3 operator/(float3 a, double b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline MOPHI_HD double3 operator/(double3 a, double3 b) {
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline MOPHI_HD void operator/=(double3& a, double3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline MOPHI_HD double3 operator/(double3 a, double b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline MOPHI_HD double3 operator/(double3 a, float b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline MOPHI_HD void operator/=(double3& a, double b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline MOPHI_HD float4 operator/(float4 a, double b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline MOPHI_HD double4 operator/(double4 a, float b) {
    return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline MOPHI_HD double4 operator/(double4 a, double b) {
    return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline MOPHI_HD void operator/=(double4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline MOPHI_HD double3 normalize(double3 v) {
    double invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// Lexicographic comparator
template <typename T>
inline MOPHI_HD bool lex_less(T a, T b) {
    if (a.x != b.x)
        return a.x < b.x;
    if (a.y != b.y)
        return a.y < b.y;
    return a.z < b.z;
}

// Float3 < is an element-wise comparison where x, y, z components are assigned priorities in that order.
// Must be in global namespace for std::less to pick it up.
inline MOPHI_HD bool operator<(const float3& a, const float3& b) {
    return lex_less(a, b);
}
inline MOPHI_HD bool operator<(const double3& a, const double3& b) {
    return lex_less(a, b);
}

// Assignment

template <typename T1>
inline MOPHI_HD float3 to_float3(const T1& a) {
    float3 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

template <typename T1>
inline MOPHI_HD double3 to_double3(const T1& a) {
    double3 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

template <typename T1, typename T2>
inline MOPHI_HD T2 to_real3(const T1& a) {
    T2 b;
    b.x = a.x;
    b.y = a.y;
    b.z = a.z;
    return b;
}

// Cause an error inside a kernel
#define MOPHI_ABORT_KERNEL(...) \
    {                           \
        printf(__VA_ARGS__);    \
        __threadfence();        \
        asm volatile("trap;");  \
    }

#endif
