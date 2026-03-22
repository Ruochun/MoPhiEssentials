/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Portions copyright (c) 2025, Ruochun Zhang
 * SPDX-License-Identifier: BSD-3-Clause
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

/**
 * @file HelperKernels.cuh
 * @brief CUDA vector math extensions and device-side helper functions.
 *
 * This file combines CUDA built-in type math utilities (originally from
 * helper_math.h by NVIDIA) with MoPhi-specific geometry helpers.
 *
 * Namespace strategy
 * ------------------
 * Operators on CUDA built-in types (float2, float3, int3, double3, etc.) remain
 * in the **global namespace**.  This is required for Argument-Dependent Lookup
 * (ADL): when a caller writes `float3 a + float3 b`, the compiler uses ADL to
 * search the namespace where float3 is defined (global).  Placing those
 * operators inside namespace mophi would silently break external code that
 * does not have a `using namespace mophi;` directive.  The same applies to
 * make_* constructor overloads, which are direct extensions of the CUDA
 * built-in constructor API and therefore also remain global.
 *
 * All other utility and geometry functions (dot, cross, length, normalize,
 * clamp, lerp, geometry helpers, etc.) live in **namespace mophi**.
 */

#ifndef MOPHI_HELPER_KERNELS_CUH
#define MOPHI_HELPER_KERNELS_CUH

#include "../common/Defines.hpp"
#include "../core/Real3.hpp"

#include <cstdint>
#include <type_traits>
#include <limits>
#include <cmath>

// ============================================================================
// CUDA-only section: requires cuda_runtime.h (compiled by nvcc only)
// ============================================================================
#ifdef __CUDACC__

    #if defined(_WIN32) || defined(_WIN64)
        #undef max
        #undef min
        #undef strtok_r
    #endif

    #include "cuda_runtime.h"

    #ifndef EXIT_WAIVED
        #define EXIT_WAIVED 2
    #endif

using uint = unsigned int;
using ushort = unsigned short;

// ============================================================================
// make_* constructor overloads (global namespace - extend CUDA built-in API)
// ============================================================================

inline MOPHI_HD float2 make_float2(float s) { return make_float2(s, s); }
inline MOPHI_HD float2 make_float2(float3 a) { return make_float2(a.x, a.y); }
inline MOPHI_HD float2 make_float2(int2 a) { return make_float2(float(a.x), float(a.y)); }
inline MOPHI_HD float2 make_float2(uint2 a) { return make_float2(float(a.x), float(a.y)); }

inline MOPHI_HD int2 make_int2(int s) { return make_int2(s, s); }
inline MOPHI_HD int2 make_int2(int3 a) { return make_int2(a.x, a.y); }
inline MOPHI_HD int2 make_int2(uint2 a) { return make_int2(int(a.x), int(a.y)); }
inline MOPHI_HD int2 make_int2(float2 a) { return make_int2(int(a.x), int(a.y)); }

inline MOPHI_HD uint2 make_uint2(uint s) { return make_uint2(s, s); }
inline MOPHI_HD uint2 make_uint2(uint3 a) { return make_uint2(a.x, a.y); }
inline MOPHI_HD uint2 make_uint2(int2 a) { return make_uint2(uint(a.x), uint(a.y)); }

inline MOPHI_HD float3 make_float3(float s) { return make_float3(s, s, s); }
inline MOPHI_HD float3 make_float3(float2 a) { return make_float3(a.x, a.y, 0.0f); }
inline MOPHI_HD float3 make_float3(float2 a, float s) { return make_float3(a.x, a.y, s); }
inline MOPHI_HD float3 make_float3(float4 a) { return make_float3(a.x, a.y, a.z); }
inline MOPHI_HD float3 make_float3(int3 a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline MOPHI_HD float3 make_float3(uint3 a) { return make_float3(float(a.x), float(a.y), float(a.z)); }

inline MOPHI_HD int3 make_int3(int s) { return make_int3(s, s, s); }
inline MOPHI_HD int3 make_int3(int2 a) { return make_int3(a.x, a.y, 0); }
inline MOPHI_HD int3 make_int3(int2 a, int s) { return make_int3(a.x, a.y, s); }
inline MOPHI_HD int3 make_int3(uint3 a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline MOPHI_HD int3 make_int3(float3 a) { return make_int3(int(a.x), int(a.y), int(a.z)); }

inline MOPHI_HD uint3 make_uint3(uint s) { return make_uint3(s, s, s); }
inline MOPHI_HD uint3 make_uint3(uint2 a) { return make_uint3(a.x, a.y, 0); }
inline MOPHI_HD uint3 make_uint3(uint2 a, uint s) { return make_uint3(a.x, a.y, s); }
inline MOPHI_HD uint3 make_uint3(uint4 a) { return make_uint3(a.x, a.y, a.z); }
inline MOPHI_HD uint3 make_uint3(int3 a) { return make_uint3(uint(a.x), uint(a.y), uint(a.z)); }

inline MOPHI_HD float4 make_float4(float s) { return make_float4(s, s, s, s); }
inline MOPHI_HD float4 make_float4(float3 a) { return make_float4(a.x, a.y, a.z, 0.0f); }
inline MOPHI_HD float4 make_float4(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }
inline MOPHI_HD float4 make_float4(int4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline MOPHI_HD float4 make_float4(uint4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline MOPHI_HD int4 make_int4(int s) { return make_int4(s, s, s, s); }
inline MOPHI_HD int4 make_int4(int3 a) { return make_int4(a.x, a.y, a.z, 0); }
inline MOPHI_HD int4 make_int4(int3 a, int w) { return make_int4(a.x, a.y, a.z, w); }
inline MOPHI_HD int4 make_int4(uint4 a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline MOPHI_HD int4 make_int4(float4 a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }

inline MOPHI_HD uint4 make_uint4(uint s) { return make_uint4(s, s, s, s); }
inline MOPHI_HD uint4 make_uint4(uint3 a) { return make_uint4(a.x, a.y, a.z, 0); }
inline MOPHI_HD uint4 make_uint4(uint3 a, uint w) { return make_uint4(a.x, a.y, a.z, w); }
inline MOPHI_HD uint4 make_uint4(int4 a) { return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w)); }

////////////////////////////////////////////////////////////////////////////////
// Operator overloads for CUDA built-in types (global namespace - required for ADL)
////////////////////////////////////////////////////////////////////////////////

// negate
inline MOPHI_HD float2 operator-(float2& a) { return make_float2(-a.x, -a.y); }
inline MOPHI_HD int2   operator-(int2& a)   { return make_int2(-a.x, -a.y); }
inline MOPHI_HD float3 operator-(float3& a) { return make_float3(-a.x, -a.y, -a.z); }
inline MOPHI_HD int3   operator-(int3& a)   { return make_int3(-a.x, -a.y, -a.z); }
inline MOPHI_HD float4 operator-(float4& a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }
inline MOPHI_HD int4   operator-(int4& a)   { return make_int4(-a.x, -a.y, -a.z, -a.w); }

// addition
inline MOPHI_HD float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline MOPHI_HD void   operator+=(float2& a, float2 b) { a.x += b.x; a.y += b.y; }
inline MOPHI_HD float2 operator+(float2 a, float b)  { return make_float2(a.x + b, a.y + b); }
inline MOPHI_HD float2 operator+(float b, float2 a)  { return make_float2(a.x + b, a.y + b); }
inline MOPHI_HD void   operator+=(float2& a, float b) { a.x += b; a.y += b; }

inline MOPHI_HD int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
inline MOPHI_HD void operator+=(int2& a, int2 b) { a.x += b.x; a.y += b.y; }
inline MOPHI_HD int2 operator+(int2 a, int b)  { return make_int2(a.x + b, a.y + b); }
inline MOPHI_HD int2 operator+(int b, int2 a)  { return make_int2(a.x + b, a.y + b); }
inline MOPHI_HD void operator+=(int2& a, int b) { a.x += b; a.y += b; }

inline MOPHI_HD uint2 operator+(uint2 a, uint2 b) { return make_uint2(a.x + b.x, a.y + b.y); }
inline MOPHI_HD void  operator+=(uint2& a, uint2 b) { a.x += b.x; a.y += b.y; }
inline MOPHI_HD uint2 operator+(uint2 a, uint b)  { return make_uint2(a.x + b, a.y + b); }
inline MOPHI_HD uint2 operator+(uint b, uint2 a)  { return make_uint2(a.x + b, a.y + b); }
inline MOPHI_HD void  operator+=(uint2& a, uint b) { a.x += b; a.y += b; }

inline MOPHI_HD float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline MOPHI_HD void   operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline MOPHI_HD float3 operator+(float3 a, float b)  { return make_float3(a.x+b, a.y+b, a.z+b); }
inline MOPHI_HD void   operator+=(float3& a, float b) { a.x += b; a.y += b; a.z += b; }
inline MOPHI_HD float3 operator+(float b, float3 a)  { return make_float3(a.x+b, a.y+b, a.z+b); }

inline MOPHI_HD int3 operator+(int3 a, int3 b) { return make_int3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline MOPHI_HD void operator+=(int3& a, int3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline MOPHI_HD int3 operator+(int3 a, int b)  { return make_int3(a.x+b, a.y+b, a.z+b); }
inline MOPHI_HD void operator+=(int3& a, int b) { a.x += b; a.y += b; a.z += b; }
inline MOPHI_HD int3 operator+(int b, int3 a)  { return make_int3(a.x+b, a.y+b, a.z+b); }

inline MOPHI_HD uint3 operator+(uint3 a, uint3 b) { return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline MOPHI_HD void  operator+=(uint3& a, uint3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline MOPHI_HD uint3 operator+(uint3 a, uint b)  { return make_uint3(a.x+b, a.y+b, a.z+b); }
inline MOPHI_HD void  operator+=(uint3& a, uint b) { a.x += b; a.y += b; a.z += b; }
inline MOPHI_HD uint3 operator+(uint b, uint3 a)  { return make_uint3(a.x+b, a.y+b, a.z+b); }

inline MOPHI_HD float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
inline MOPHI_HD void   operator+=(float4& a, float4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; }
inline MOPHI_HD float4 operator+(float4 a, float b)  { return make_float4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD float4 operator+(float b, float4 a)  { return make_float4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD void   operator+=(float4& a, float b) { a.x+=b; a.y+=b; a.z+=b; a.w+=b; }

inline MOPHI_HD int4 operator+(int4 a, int4 b) { return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline MOPHI_HD void operator+=(int4& a, int4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; }
inline MOPHI_HD int4 operator+(int4 a, int b)  { return make_int4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD int4 operator+(int b, int4 a)  { return make_int4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD void operator+=(int4& a, int b) { a.x+=b; a.y+=b; a.z+=b; a.w+=b; }

inline MOPHI_HD uint4 operator+(uint4 a, uint4 b) { return make_uint4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline MOPHI_HD void  operator+=(uint4& a, uint4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; }
inline MOPHI_HD uint4 operator+(uint4 a, uint b)  { return make_uint4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD uint4 operator+(uint b, uint4 a)  { return make_uint4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline MOPHI_HD void  operator+=(uint4& a, uint b) { a.x+=b; a.y+=b; a.z+=b; a.w+=b; }

// subtract
inline MOPHI_HD float2 operator-(float2 a, float2 b) { return make_float2(a.x-b.x, a.y-b.y); }
inline MOPHI_HD void   operator-=(float2& a, float2 b) { a.x -= b.x; a.y -= b.y; }
inline MOPHI_HD float2 operator-(float2 a, float b)  { return make_float2(a.x-b, a.y-b); }
inline MOPHI_HD float2 operator-(float b, float2 a)  { return make_float2(b-a.x, b-a.y); }
inline MOPHI_HD void   operator-=(float2& a, float b) { a.x -= b; a.y -= b; }

inline MOPHI_HD int2 operator-(int2 a, int2 b) { return make_int2(a.x-b.x, a.y-b.y); }
inline MOPHI_HD void operator-=(int2& a, int2 b) { a.x -= b.x; a.y -= b.y; }
inline MOPHI_HD int2 operator-(int2 a, int b)  { return make_int2(a.x-b, a.y-b); }
inline MOPHI_HD int2 operator-(int b, int2 a)  { return make_int2(b-a.x, b-a.y); }
inline MOPHI_HD void operator-=(int2& a, int b) { a.x -= b; a.y -= b; }

inline MOPHI_HD uint2 operator-(uint2 a, uint2 b) { return make_uint2(a.x-b.x, a.y-b.y); }
inline MOPHI_HD void  operator-=(uint2& a, uint2 b) { a.x -= b.x; a.y -= b.y; }
inline MOPHI_HD uint2 operator-(uint2 a, uint b)  { return make_uint2(a.x-b, a.y-b); }
inline MOPHI_HD uint2 operator-(uint b, uint2 a)  { return make_uint2(b-a.x, b-a.y); }
inline MOPHI_HD void  operator-=(uint2& a, uint b) { a.x -= b; a.y -= b; }

inline MOPHI_HD float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline MOPHI_HD void   operator-=(float3& a, float3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD float3 operator-(float3 a, float b)  { return make_float3(a.x-b, a.y-b, a.z-b); }
inline MOPHI_HD float3 operator-(float b, float3 a)  { return make_float3(b-a.x, b-a.y, b-a.z); }
inline MOPHI_HD void   operator-=(float3& a, float b) { a.x-=b; a.y-=b; a.z-=b; }

inline MOPHI_HD int3 operator-(int3 a, int3 b) { return make_int3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline MOPHI_HD void operator-=(int3& a, int3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD int3 operator-(int3 a, int b)  { return make_int3(a.x-b, a.y-b, a.z-b); }
inline MOPHI_HD int3 operator-(int b, int3 a)  { return make_int3(b-a.x, b-a.y, b-a.z); }
inline MOPHI_HD void operator-=(int3& a, int b) { a.x-=b; a.y-=b; a.z-=b; }

inline MOPHI_HD uint3 operator-(uint3 a, uint3 b) { return make_uint3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline MOPHI_HD void  operator-=(uint3& a, uint3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD uint3 operator-(uint3 a, uint b)  { return make_uint3(a.x-b, a.y-b, a.z-b); }
inline MOPHI_HD uint3 operator-(uint b, uint3 a)  { return make_uint3(b-a.x, b-a.y, b-a.z); }
inline MOPHI_HD void  operator-=(uint3& a, uint b) { a.x-=b; a.y-=b; a.z-=b; }

inline MOPHI_HD float4 operator-(float4 a, float4 b) { return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline MOPHI_HD void   operator-=(float4& a, float4 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w; }
inline MOPHI_HD float4 operator-(float4 a, float b)  { return make_float4(a.x-b, a.y-b, a.z-b, a.w-b); }
inline MOPHI_HD void   operator-=(float4& a, float b) { a.x-=b; a.y-=b; a.z-=b; a.w-=b; }

inline MOPHI_HD int4 operator-(int4 a, int4 b) { return make_int4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline MOPHI_HD void operator-=(int4& a, int4 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w; }
inline MOPHI_HD int4 operator-(int4 a, int b)  { return make_int4(a.x-b, a.y-b, a.z-b, a.w-b); }
inline MOPHI_HD int4 operator-(int b, int4 a)  { return make_int4(b-a.x, b-a.y, b-a.z, b-a.w); }
inline MOPHI_HD void operator-=(int4& a, int b) { a.x-=b; a.y-=b; a.z-=b; a.w-=b; }

inline MOPHI_HD uint4 operator-(uint4 a, uint4 b) { return make_uint4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline MOPHI_HD void  operator-=(uint4& a, uint4 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w; }
inline MOPHI_HD uint4 operator-(uint4 a, uint b)  { return make_uint4(a.x-b, a.y-b, a.z-b, a.w-b); }
inline MOPHI_HD uint4 operator-(uint b, uint4 a)  { return make_uint4(b-a.x, b-a.y, b-a.z, b-a.w); }
inline MOPHI_HD void  operator-=(uint4& a, uint b) { a.x-=b; a.y-=b; a.z-=b; a.w-=b; }

// multiply
inline MOPHI_HD float2 operator*(float2 a, float2 b) { return make_float2(a.x*b.x, a.y*b.y); }
inline MOPHI_HD void   operator*=(float2& a, float2 b) { a.x*=b.x; a.y*=b.y; }
inline MOPHI_HD float2 operator*(float2 a, float b)  { return make_float2(a.x*b, a.y*b); }
inline MOPHI_HD float2 operator*(float b, float2 a)  { return make_float2(b*a.x, b*a.y); }
inline MOPHI_HD void   operator*=(float2& a, float b) { a.x*=b; a.y*=b; }

inline MOPHI_HD int2 operator*(int2 a, int2 b) { return make_int2(a.x*b.x, a.y*b.y); }
inline MOPHI_HD void operator*=(int2& a, int2 b) { a.x*=b.x; a.y*=b.y; }
inline MOPHI_HD int2 operator*(int2 a, int b)  { return make_int2(a.x*b, a.y*b); }
inline MOPHI_HD int2 operator*(int b, int2 a)  { return make_int2(b*a.x, b*a.y); }
inline MOPHI_HD void operator*=(int2& a, int b) { a.x*=b; a.y*=b; }

inline MOPHI_HD uint2 operator*(uint2 a, uint2 b) { return make_uint2(a.x*b.x, a.y*b.y); }
inline MOPHI_HD void  operator*=(uint2& a, uint2 b) { a.x*=b.x; a.y*=b.y; }
inline MOPHI_HD uint2 operator*(uint2 a, uint b)  { return make_uint2(a.x*b, a.y*b); }
inline MOPHI_HD uint2 operator*(uint b, uint2 a)  { return make_uint2(b*a.x, b*a.y); }
inline MOPHI_HD void  operator*=(uint2& a, uint b) { a.x*=b; a.y*=b; }

inline MOPHI_HD float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline MOPHI_HD void   operator*=(float3& a, float3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; }
inline MOPHI_HD float3 operator*(float3 a, float b)  { return make_float3(a.x*b, a.y*b, a.z*b); }
inline MOPHI_HD float3 operator*(float b, float3 a)  { return make_float3(b*a.x, b*a.y, b*a.z); }
inline MOPHI_HD void   operator*=(float3& a, float b) { a.x*=b; a.y*=b; a.z*=b; }

inline MOPHI_HD int3 operator*(int3 a, int3 b) { return make_int3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline MOPHI_HD void operator*=(int3& a, int3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; }
inline MOPHI_HD int3 operator*(int3 a, int b)  { return make_int3(a.x*b, a.y*b, a.z*b); }
inline MOPHI_HD int3 operator*(int b, int3 a)  { return make_int3(b*a.x, b*a.y, b*a.z); }
inline MOPHI_HD void operator*=(int3& a, int b) { a.x*=b; a.y*=b; a.z*=b; }

inline MOPHI_HD uint3 operator*(uint3 a, uint3 b) { return make_uint3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline MOPHI_HD void  operator*=(uint3& a, uint3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; }
inline MOPHI_HD uint3 operator*(uint3 a, uint b)  { return make_uint3(a.x*b, a.y*b, a.z*b); }
inline MOPHI_HD uint3 operator*(uint b, uint3 a)  { return make_uint3(b*a.x, b*a.y, b*a.z); }
inline MOPHI_HD void  operator*=(uint3& a, uint b) { a.x*=b; a.y*=b; a.z*=b; }

inline MOPHI_HD float4 operator*(float4 a, float4 b) { return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
inline MOPHI_HD void   operator*=(float4& a, float4 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w; }
inline MOPHI_HD float4 operator*(float4 a, float b)  { return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }
inline MOPHI_HD float4 operator*(float b, float4 a)  { return make_float4(b*a.x, b*a.y, b*a.z, b*a.w); }
inline MOPHI_HD void   operator*=(float4& a, float b) { a.x*=b; a.y*=b; a.z*=b; a.w*=b; }

inline MOPHI_HD int4 operator*(int4 a, int4 b) { return make_int4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
inline MOPHI_HD void operator*=(int4& a, int4 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w; }
inline MOPHI_HD int4 operator*(int4 a, int b)  { return make_int4(a.x*b, a.y*b, a.z*b, a.w*b); }
inline MOPHI_HD int4 operator*(int b, int4 a)  { return make_int4(b*a.x, b*a.y, b*a.z, b*a.w); }
inline MOPHI_HD void operator*=(int4& a, int b) { a.x*=b; a.y*=b; a.z*=b; a.w*=b; }

inline MOPHI_HD uint4 operator*(uint4 a, uint4 b) { return make_uint4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
inline MOPHI_HD void  operator*=(uint4& a, uint4 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w; }
inline MOPHI_HD uint4 operator*(uint4 a, uint b)  { return make_uint4(a.x*b, a.y*b, a.z*b, a.w*b); }
inline MOPHI_HD uint4 operator*(uint b, uint4 a)  { return make_uint4(b*a.x, b*a.y, b*a.z, b*a.w); }
inline MOPHI_HD void  operator*=(uint4& a, uint b) { a.x*=b; a.y*=b; a.z*=b; a.w*=b; }

// divide
inline MOPHI_HD float2 operator/(float2 a, float2 b) { return make_float2(a.x/b.x, a.y/b.y); }
inline MOPHI_HD void   operator/=(float2& a, float2 b) { a.x/=b.x; a.y/=b.y; }
inline MOPHI_HD float2 operator/(float2 a, float b)  { return make_float2(a.x/b, a.y/b); }
inline MOPHI_HD void   operator/=(float2& a, float b) { a.x/=b; a.y/=b; }
inline MOPHI_HD float2 operator/(float b, float2 a)  { return make_float2(b/a.x, b/a.y); }

inline MOPHI_HD float3 operator/(float3 a, float3 b) { return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }
inline MOPHI_HD void   operator/=(float3& a, float3 b) { a.x/=b.x; a.y/=b.y; a.z/=b.z; }
inline MOPHI_HD float3 operator/(float3 a, float b)  { return make_float3(a.x/b, a.y/b, a.z/b); }
inline MOPHI_HD void   operator/=(float3& a, float b) { a.x/=b; a.y/=b; a.z/=b; }
inline MOPHI_HD float3 operator/(float b, float3 a)  { return make_float3(b/a.x, b/a.y, b/a.z); }

inline MOPHI_HD float4 operator/(float4 a, float4 b) { return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }
inline MOPHI_HD void   operator/=(float4& a, float4 b) { a.x/=b.x; a.y/=b.y; a.z/=b.z; a.w/=b.w; }
inline MOPHI_HD float4 operator/(float4 a, float b)  { return make_float4(a.x/b, a.y/b, a.z/b, a.w/b); }
inline MOPHI_HD void   operator/=(float4& a, float b) { a.x/=b; a.y/=b; a.z/=b; a.w/=b; }
inline MOPHI_HD float4 operator/(float b, float4 a)  { return make_float4(b/a.x, b/a.y, b/a.z, b/a.w); }

// double3 / double4 operators (added by Ruochun)
inline MOPHI_HD double3 operator+(double3 a, double3 b) { return make_double3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline MOPHI_HD double3 operator-(double3 a, double3 b) { return make_double3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline MOPHI_HD float3  operator+(double3 a, float3 b)  { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline MOPHI_HD float3  operator-(double3 a, float3 b)  { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline MOPHI_HD void operator+=(double3& a, double3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; }
inline MOPHI_HD void operator-=(double3& a, double3 b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD void operator+=(double3& a, double b)  { a.x+=b; a.y+=b; a.z+=b; }
inline MOPHI_HD void operator-=(double3& a, double b)  { a.x-=b; a.y-=b; a.z-=b; }
inline MOPHI_HD void operator+=(float3& a, double3 b)  { a.x+=b.x; a.y+=b.y; a.z+=b.z; }
inline MOPHI_HD void operator-=(float3& a, double3 b)  { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD void operator+=(float3& a, double b)   { a.x+=b; a.y+=b; a.z+=b; }
inline MOPHI_HD void operator-=(float3& a, double b)   { a.x-=b; a.y-=b; a.z-=b; }
inline MOPHI_HD void operator+=(double3& a, float3 b)  { a.x+=b.x; a.y+=b.y; a.z+=b.z; }
inline MOPHI_HD void operator-=(double3& a, float3 b)  { a.x-=b.x; a.y-=b.y; a.z-=b.z; }
inline MOPHI_HD void operator+=(double3& a, float b)   { a.x+=b; a.y+=b; a.z+=b; }
inline MOPHI_HD void operator-=(double3& a, float b)   { a.x-=b; a.y-=b; a.z-=b; }
inline MOPHI_HD double3 operator*(double3 a, double3 b) { return make_double3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline MOPHI_HD void    operator*=(double3& a, double3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; }
inline MOPHI_HD double3 operator*(double3 a, double b)  { return make_double3(a.x*b, a.y*b, a.z*b); }
inline MOPHI_HD double3 operator*(double b, double3 a)  { return make_double3(b*a.x, b*a.y, b*a.z); }
inline MOPHI_HD void    operator*=(double3& a, double b) { a.x*=b; a.y*=b; a.z*=b; }
inline MOPHI_HD float3  operator/(float3 a, double b)   { return make_float3(a.x/b, a.y/b, a.z/b); }
inline MOPHI_HD double3 operator/(double3 a, double3 b) { return make_double3(a.x/b.x, a.y/b.y, a.z/b.z); }
inline MOPHI_HD void    operator/=(double3& a, double3 b) { a.x/=b.x; a.y/=b.y; a.z/=b.z; }
inline MOPHI_HD double3 operator/(double3 a, double b)  { return make_double3(a.x/b, a.y/b, a.z/b); }
inline MOPHI_HD double3 operator/(double3 a, float b)   { return make_double3(a.x/b, a.y/b, a.z/b); }
inline MOPHI_HD void    operator/=(double3& a, double b) { a.x/=b; a.y/=b; a.z/=b; }
inline MOPHI_HD float4  operator/(float4 a, double b)   { return make_float4(a.x/b, a.y/b, a.z/b, a.w/b); }
inline MOPHI_HD double4 operator/(double4 a, float b)   { return make_double4(a.x/b, a.y/b, a.z/b, a.w/b); }
inline MOPHI_HD double4 operator/(double4 a, double b)  { return make_double4(a.x/b, a.y/b, a.z/b, a.w/b); }
inline MOPHI_HD void    operator/=(double4& a, float b)  { a.x/=b; a.y/=b; a.z/=b; a.w/=b; }

// Lexicographic operator< for float3/double3 - must be in global namespace for std::less to pick it up.
inline MOPHI_HD bool operator<(const float3& a, const float3& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}
inline MOPHI_HD bool operator<(const double3& a, const double3& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

// Cause an error inside a kernel (macro - not namespace-bound)
#define MOPHI_ABORT_KERNEL(...)  \
    {                            \
        printf(__VA_ARGS__);     \
        __threadfence();         \
        asm volatile("trap;");   \
    }

// ============================================================================
// Math utility functions (namespace mophi)
// ============================================================================
namespace mophi {

// min/max for CUDA vector types
inline MOPHI_HD float2 fminf(float2 a, float2 b) { return make_float2(fminf(a.x,b.x), fminf(a.y,b.y)); }
inline MOPHI_HD float3 fminf(float3 a, float3 b) {
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline MOPHI_HD float4 fminf(float4 a, float4 b) {
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
inline MOPHI_HD int2  min(int2 a, int2 b)  { return make_int2(min(a.x,b.x), min(a.y,b.y)); }
inline MOPHI_HD int3  min(int3 a, int3 b)  { return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
inline MOPHI_HD int4  min(int4 a, int4 b)  { return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)); }
inline MOPHI_HD uint2 min(uint2 a, uint2 b) { return make_uint2(min(a.x,b.x), min(a.y,b.y)); }
inline MOPHI_HD uint3 min(uint3 a, uint3 b) { return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
inline MOPHI_HD uint4 min(uint4 a, uint4 b) {
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}
inline MOPHI_HD float2 fmaxf(float2 a, float2 b) { return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y)); }
inline MOPHI_HD float3 fmaxf(float3 a, float3 b) {
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline MOPHI_HD float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
inline MOPHI_HD int2  max(int2 a, int2 b)  { return make_int2(max(a.x,b.x), max(a.y,b.y)); }
inline MOPHI_HD int3  max(int3 a, int3 b)  { return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
inline MOPHI_HD int4  max(int4 a, int4 b)  { return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)); }
inline MOPHI_HD uint2 max(uint2 a, uint2 b) { return make_uint2(max(a.x,b.x), max(a.y,b.y)); }
inline MOPHI_HD uint3 max(uint3 a, uint3 b) { return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
inline MOPHI_HD uint4 max(uint4 a, uint4 b) {
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

// logical helpers
inline MOPHI_HD bool float_near(const float& a, const float& b, float tol = 1e-6f) {
    return fabs(a - b) < tol;
}
inline MOPHI_HD bool float3_near(const float3& a, const float3& b, float tol = 1e-6f) {
    return float_near(a.x, b.x, tol) && float_near(a.y, b.y, tol) && float_near(a.z, b.z, tol);
}

// dot product
inline MOPHI_HD float  dot(float2 a, float2 b) { return a.x*b.x + a.y*b.y; }
inline MOPHI_HD float  dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline MOPHI_HD float  dot(float4 a, float4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline MOPHI_HD int    dot(int2 a, int2 b)     { return a.x*b.x + a.y*b.y; }
inline MOPHI_HD int    dot(int3 a, int3 b)     { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline MOPHI_HD int    dot(int4 a, int4 b)     { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline MOPHI_HD uint   dot(uint2 a, uint2 b)   { return a.x*b.x + a.y*b.y; }
inline MOPHI_HD uint   dot(uint3 a, uint3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline MOPHI_HD uint   dot(uint4 a, uint4 b)   { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline MOPHI_HD double dot(double3 a, double3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline MOPHI_HD float  dot(double3 a, float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline MOPHI_HD double dot(double4 a, double4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline MOPHI_HD float  dot(double4 a, float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

// length
inline MOPHI_HD float  length(float2 v) { return sqrtf(dot(v, v)); }
inline MOPHI_HD float  length(float3 v) { return sqrtf(dot(v, v)); }
inline MOPHI_HD float  length(float4 v) { return sqrtf(dot(v, v)); }
inline MOPHI_HD double length(double3 v) { return sqrt(dot(v, v)); }
inline MOPHI_HD double length(double4 v) { return sqrt(dot(v, v)); }

// normalize
inline MOPHI_HD float2  normalize(float2 v) { return v * rsqrtf(dot(v, v)); }
inline MOPHI_HD float3  normalize(float3 v) { return v * rsqrtf(dot(v, v)); }
inline MOPHI_HD float4  normalize(float4 v) { return v * rsqrtf(dot(v, v)); }
inline MOPHI_HD double3 normalize(double3 v) { return v * (1.0 / length(v)); }

// cross product
inline MOPHI_HD float3  cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline MOPHI_HD double3 cross(double3 a, double3 b) {
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// lerp: linear interpolation between a and b for t in [0, 1]
inline MOPHI_HD float  lerp(float a, float b, float t)   { return a + t*(b-a); }
inline MOPHI_HD float2 lerp(float2 a, float2 b, float t) { return a + t*(b-a); }
inline MOPHI_HD float3 lerp(float3 a, float3 b, float t) { return a + t*(b-a); }
inline MOPHI_HD float4 lerp(float4 a, float4 b, float t) { return a + t*(b-a); }

// clamp: clamp v to [a, b]
inline MOPHI_HD float  clamp(float f, float a, float b)  { return fmaxf(a, fminf(f, b)); }
inline MOPHI_HD int    clamp(int f, int a, int b)         { return max(a, min(f, b)); }
inline MOPHI_HD uint   clamp(uint f, uint a, uint b)      { return max(a, min(f, b)); }
inline MOPHI_HD float2 clamp(float2 v, float a, float b) { return make_float2(clamp(v.x,a,b), clamp(v.y,a,b)); }
inline MOPHI_HD float2 clamp(float2 v, float2 a, float2 b) {
    return make_float2(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y));
}
inline MOPHI_HD float3 clamp(float3 v, float a, float b) {
    return make_float3(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b));
}
inline MOPHI_HD float3 clamp(float3 v, float3 a, float3 b) {
    return make_float3(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z));
}
inline MOPHI_HD float4 clamp(float4 v, float a, float b) {
    return make_float4(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b), clamp(v.w,a,b));
}
inline MOPHI_HD float4 clamp(float4 v, float4 a, float4 b) {
    return make_float4(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z), clamp(v.w,a.w,b.w));
}
inline MOPHI_HD int2 clamp(int2 v, int a, int b) { return make_int2(clamp(v.x,a,b), clamp(v.y,a,b)); }
inline MOPHI_HD int2 clamp(int2 v, int2 a, int2 b) {
    return make_int2(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y));
}
inline MOPHI_HD int3 clamp(int3 v, int a, int b) {
    return make_int3(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b));
}
inline MOPHI_HD int3 clamp(int3 v, int3 a, int3 b) {
    return make_int3(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z));
}
inline MOPHI_HD int4 clamp(int4 v, int a, int b) {
    return make_int4(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b), clamp(v.w,a,b));
}
inline MOPHI_HD int4 clamp(int4 v, int4 a, int4 b) {
    return make_int4(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z), clamp(v.w,a.w,b.w));
}
inline MOPHI_HD uint2 clamp(uint2 v, uint a, uint b) { return make_uint2(clamp(v.x,a,b), clamp(v.y,a,b)); }
inline MOPHI_HD uint2 clamp(uint2 v, uint2 a, uint2 b) {
    return make_uint2(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y));
}
inline MOPHI_HD uint3 clamp(uint3 v, uint a, uint b) {
    return make_uint3(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b));
}
inline MOPHI_HD uint3 clamp(uint3 v, uint3 a, uint3 b) {
    return make_uint3(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z));
}
inline MOPHI_HD uint4 clamp(uint4 v, uint a, uint b) {
    return make_uint4(clamp(v.x,a,b), clamp(v.y,a,b), clamp(v.z,a,b), clamp(v.w,a,b));
}
inline MOPHI_HD uint4 clamp(uint4 v, uint4 a, uint4 b) {
    return make_uint4(clamp(v.x,a.x,b.x), clamp(v.y,a.y,b.y), clamp(v.z,a.z,b.z), clamp(v.w,a.w,b.w));
}

// floor
inline MOPHI_HD float2 floorf(float2 v) { return make_float2(floorf(v.x), floorf(v.y)); }
inline MOPHI_HD float3 floorf(float3 v) { return make_float3(floorf(v.x), floorf(v.y), floorf(v.z)); }
inline MOPHI_HD float4 floorf(float4 v) { return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)); }

// frac: fractional portion
inline MOPHI_HD float  fracf(float v)  { return v - floorf(v); }
inline MOPHI_HD float2 fracf(float2 v) { return make_float2(fracf(v.x), fracf(v.y)); }
inline MOPHI_HD float3 fracf(float3 v) { return make_float3(fracf(v.x), fracf(v.y), fracf(v.z)); }
inline MOPHI_HD float4 fracf(float4 v) { return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w)); }

// fmod
inline MOPHI_HD float2 fmodf(float2 a, float2 b) { return make_float2(fmodf(a.x,b.x), fmodf(a.y,b.y)); }
inline MOPHI_HD float3 fmodf(float3 a, float3 b) {
    return make_float3(fmodf(a.x,b.x), fmodf(a.y,b.y), fmodf(a.z,b.z));
}
inline MOPHI_HD float4 fmodf(float4 a, float4 b) {
    return make_float4(fmodf(a.x,b.x), fmodf(a.y,b.y), fmodf(a.z,b.z), fmodf(a.w,b.w));
}

// absolute value
inline MOPHI_HD float2 fabs(float2 v) { return make_float2(fabs(v.x), fabs(v.y)); }
inline MOPHI_HD float3 fabs(float3 v) { return make_float3(fabs(v.x), fabs(v.y), fabs(v.z)); }
inline MOPHI_HD float4 fabs(float4 v) { return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w)); }
inline MOPHI_HD int2   abs(int2 v)    { return make_int2(abs(v.x), abs(v.y)); }
inline MOPHI_HD int3   abs(int3 v)    { return make_int3(abs(v.x), abs(v.y), abs(v.z)); }
inline MOPHI_HD int4   abs(int4 v)    { return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w)); }

// reflect: reflection of incident ray I around surface normal N (N should be normalized)
inline MOPHI_HD float3 reflect(float3 i, float3 n) { return i - 2.0f * n * dot(n, i); }

// smoothstep: returns 0 if x < a, 1 if x > b, else smooth interpolation in [0,1]
inline MOPHI_HD float smoothstep(float a, float b, float x) {
    float y = clamp((x-a)/(b-a), 0.0f, 1.0f);
    return y*y*(3.0f - 2.0f*y);
}
inline MOPHI_HD float2 smoothstep(float2 a, float2 b, float2 x) {
    float2 y = clamp((x-a)/(b-a), 0.0f, 1.0f);
    return y*y*(make_float2(3.0f) - make_float2(2.0f)*y);
}
inline MOPHI_HD float3 smoothstep(float3 a, float3 b, float3 x) {
    float3 y = clamp((x-a)/(b-a), 0.0f, 1.0f);
    return y*y*(make_float3(3.0f) - make_float3(2.0f)*y);
}
inline MOPHI_HD float4 smoothstep(float4 a, float4 b, float4 x) {
    float4 y = clamp((x-a)/(b-a), 0.0f, 1.0f);
    return y*y*(make_float4(3.0f) - make_float4(2.0f)*y);
}

// Conversion helpers
template <typename T1>
inline MOPHI_HD float3 to_float3(const T1& a) {
    float3 b; b.x = a.x; b.y = a.y; b.z = a.z; return b;
}
template <typename T1>
inline MOPHI_HD double3 to_double3(const T1& a) {
    double3 b; b.x = a.x; b.y = a.y; b.z = a.z; return b;
}
template <typename T1, typename T2>
inline MOPHI_HD T2 to_real3(const T1& a) {
    T2 b; b.x = a.x; b.y = a.y; b.z = a.z; return b;
}

}  // namespace mophi

#endif  // __CUDACC__

// ============================================================================
// Geometry helpers (namespace mophi) - available with or without CUDA
// ============================================================================
namespace mophi {

// Sign function
template <typename T1>
MOPHI_HD inline int sgn(const T1& val) {
    return (T1(0) < val) - (val < T1(0));
}

// Integer division that rounds towards -infty
template <typename T1, typename T2>
MOPHI_HD inline T1 div_floor(const T1& a, const T2& b) {
    T1 res = a / b;
    T1 rem = a % b;
    // Correct division result downwards if up-rounding happened,
    // (for non-zero remainder of sign different than the divisor).
    T1 corr = (rem != 0 && ((rem < 0) != (b < 0)));
    return res - corr;
}

// Modulus that rounds towards -infty
template <typename T1, typename T2>
MOPHI_HD inline T1 mod_floor(const T1& a, const T2& b) {
    if (b < 0)
        return -mod_floor(-a, -b);
    T1 ret = a % b;
    if (ret < 0)
        ret += b;
    return ret;
}

template <typename T>
MOPHI_HD inline T tri_centroid(const T& A, const T& B, const T& C) {
    return {(A.x() + B.x() + C.x()) / 3, (A.y() + B.y() + C.y()) / 3, (A.z() + B.z() + C.z()) / 3};
}
template <typename T>
MOPHI_HD inline T tet_centroid(const T& a, const T& b, const T& c, const T& d) {
    return {(a.x() + b.x() + c.x() + d.x()) / 4, (a.y() + b.y() + c.y() + d.y()) / 4,
            (a.z() + b.z() + c.z() + d.z()) / 4};
}
template <typename T>
MOPHI_HD inline T tet_volume(const Real3<T>& a, const Real3<T>& b, const Real3<T>& c, const Real3<T>& d) {
    return std::abs((b - a) ^ ((c - a) % (d - a))) / 6.0;  // ^ is dot, % is cross
}
template <typename T>
MOPHI_HD inline T tri_area(const Real3<T>& a, const Real3<T>& b, const Real3<T>& c) {
    return 0.5 * ((b - a) % (c - a)).Length();
}
template <typename T>
MOPHI_HD inline void tet_gradN_phys(const Real3<T>& a,
                                    const Real3<T>& b,
                                    const Real3<T>& c,
                                    const Real3<T>& d,
                                    Real3<T> g[4]) {
    // ∇N1 = (e2 × e3)/det, ∇N2 = (e3 × e1)/det, ∇N3 = (e1 × e2)/det, ∇N0 = -Σ
    const Real3<T> e1 = b - a, e2 = c - a, e3 = d - a;
    const T det = (e1 ^ (e2 % e3));
    const T inv = (T)1.0 / det;
    g[1] = (e2 % e3) * inv;
    g[2] = (e3 % e1) * inv;
    g[3] = (e1 % e2) * inv;
    g[0] = (g[1] + g[2] + g[3]) * (T)(-1);
}

// hex helpers
template <typename T>
MOPHI_HD inline T hex_centroid(const T v[8]) {
    double inv8 = 1.0 / 8.0;
    return (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]) * inv8;
}
template <typename T>
MOPHI_HD inline T hex_volume(const Real3<T> v[8]) {
    // decompose into 5 tets (fan from v0)
    const int ind[5][4] = {{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}};
    T vol = 0;
    for (int t = 0; t < 5; ++t)
        vol += tet_volume(v[ind[t][0]], v[ind[t][1]], v[ind[t][2]], v[ind[t][3]]);
    return vol;
}

// In an upper-triangular (including the diagonal part) matrix, given i and j, this function returns the index of the
// corresponding flatten-ed non-zero entries (col-major like in matlab). This function does not assume i <= j. It is
// used in locating masks that maps the contact between families.
template <typename T1>
MOPHI_HD inline T1 locate_pair(const T1& i, const T1& j) {
    if (i > j)
        return locate_mask_pair(j, i);
    return (1 + j) * j / 2 + i;
}

// Magic function that converts an index of a flatten-ed upper-triangular matrix (EXCLUDING the diagonal) to its
// corresponding i and j. It is ROW-major. It is used to map contact pair numbers in a bin.
template <typename T1>
MOPHI_HD inline void recover_pair_no_diag(T1& i, T1& j, const T1& ind, const T1& n) {
    i = n - 2 - (T1)(sqrt((float)(4 * n * (n - 1) - 7 - 8 * ind)) / 2.0 - 0.5);
    j = ind + i + 1 + (n - i) * ((n - i) - 1) / 2 - n * (n - 1) / 2;
}

}  // namespace mophi

#endif  // MOPHI_HELPER_KERNELS_CUH
