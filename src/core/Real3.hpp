//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

// This is a modification of the code by Alessandro Tasora and Radu Serban (license below)
//
// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Radu Serban
// ================================================================

#ifndef MOPHI_VECTORS_H
#define MOPHI_VECTORS_H

#include <algorithm>
#include <cmath>
#include <ostream>
#include <limits>
#include <cuda_runtime_api.h>

#ifndef MOPHI_HD
    #ifdef __CUDACC__
        #define MOPHI_HD __host__ __device__
    #else
        #define MOPHI_HD
    #endif
#endif

namespace mophi {

/// Definition of general purpose 3d vector variables, such as points in 3D.
/// This class implements the vectorial algebra in 3D (Gibbs products).
/// Real3 is templated by precision, with default 'float'.
template <class Real = float>
class alignas(sizeof(Real) * 4) Real3 {  // alignment is for using Real3 in CUDA kernels
  public:
    // CONSTRUCTORS
    MOPHI_HD Real3();
    MOPHI_HD Real3(Real x, Real y, Real z);
    MOPHI_HD Real3(Real a);
    MOPHI_HD Real3(const Real3<Real>& other);

    /// Copy constructor with type change.
    template <class RealB>
    MOPHI_HD Real3(const Real3<RealB>& other);

    /// Access to components
    MOPHI_HD Real& x() { return m_data[0]; }
    MOPHI_HD Real& y() { return m_data[1]; }
    MOPHI_HD Real& z() { return m_data[2]; }
    const MOPHI_HD Real& x() const { return m_data[0]; }
    const MOPHI_HD Real& y() const { return m_data[1]; }
    const MOPHI_HD Real& z() const { return m_data[2]; }

    /// Access to underlying array storage.
    MOPHI_HD Real* data() { return m_data; }
    const MOPHI_HD Real* data() const { return m_data; }

    // SET FUNCTIONS

    /// Set the three values of the vector at once.
    MOPHI_HD void Set(Real x, Real y, Real z);

    /// Set the vector as a copy of another vector.
    MOPHI_HD void Set(const Real3<Real>& v);

    /// Set all the vector components ts to the same scalar.
    MOPHI_HD void Set(Real s);

    /// Set the vector to the null vector.
    MOPHI_HD void SetNull();

    /// Return true if this vector is the null vector.
    MOPHI_HD bool IsNull() const;

    /// Return true if this vector is equal to another vector.
    MOPHI_HD bool Equals(const Real3<Real>& other) const;

    /// Return true if this vector is equal to another vector, within a tolerance 'tol'.
    MOPHI_HD bool Equals(const Real3<Real>& other, Real tol) const;

    // VECTOR NORMS

    /// Compute the euclidean norm of the vector, that is its length or magnitude.
    MOPHI_HD Real Length() const;

    /// Compute the squared euclidean norm of the vector.
    MOPHI_HD Real Length2() const;

    /// Compute the infinity norm of the vector, that is the maximum absolute value of one of its elements.
    MOPHI_HD Real LengthInf() const;

    // OPERATORS OVERLOADING
    //
    // Note: c++ automatically creates temporary objects to store intermediate
    // results in long formulas, such as a= b*c*d, so the usage of operators
    // may give slower results than a wise (less readable however) usage of
    // Dot(), Cross() etc.. Also pay attention to C++ operator precedence rules!

    /// Subscript operator.
    MOPHI_HD Real& operator[](unsigned index);
    const MOPHI_HD Real& operator[](unsigned index) const;

    /// Assignment operator (copy from another vector).
    MOPHI_HD Real3<Real>& operator=(const Real3<Real>& other);

    /// Assignment operator (copy from another vector) with type change.
    template <class RealB>
    MOPHI_HD Real3<Real>& operator=(const Real3<RealB>& other);

    /// Operators for sign change.
    MOPHI_HD Real3<Real> operator+() const;
    MOPHI_HD Real3<Real> operator-() const;

    /// Operator for vector sum.
    MOPHI_HD Real3<Real> operator+(const Real3<Real>& other) const;
    MOPHI_HD Real3<Real>& operator+=(const Real3<Real>& other);

    /// Operator for vector difference.
    MOPHI_HD Real3<Real> operator-(const Real3<Real>& other) const;
    MOPHI_HD Real3<Real>& operator-=(const Real3<Real>& other);

    /// Operator for element-wise multiplication.
    /// Note that this is neither dot product nor cross product.
    MOPHI_HD Real3<Real> operator*(const Real3<Real>& other) const;
    MOPHI_HD Real3<Real>& operator*=(const Real3<Real>& other);

    /// Operator for element-wise division.
    /// Note that 3D vector algebra is a skew field, non-divisional algebra,
    /// so this division operation is just an element-by element division.
    MOPHI_HD Real3<Real> operator/(const Real3<Real>& other) const;
    MOPHI_HD Real3<Real>& operator/=(const Real3<Real>& other);

    /// Operator for scaling the vector by a scalar value, as V*s
    MOPHI_HD Real3<Real> operator*(Real s) const;
    MOPHI_HD Real3<Real>& operator*=(Real s);

    /// Operator for scaling the vector by inverse of a scalar value, as v/s
    MOPHI_HD Real3<Real> operator/(Real v) const;
    MOPHI_HD Real3<Real>& operator/=(Real v);

    /// Operator for dot product: A^B means the scalar dot-product A*B
    /// Note: pay attention to operator low precedence (see C++ precedence rules!)
    MOPHI_HD Real operator^(const Real3<Real>& other) const;

    /// Operator for cross product: A%B means the vector cross-product AxB
    /// Note: pay attention to operator low precedence (see C++ precedence rules!)
    MOPHI_HD Real3<Real> operator%(const Real3<Real>& other) const;
    MOPHI_HD Real3<Real>& operator%=(const Real3<Real>& other);

    /// Component-wise comparison operators
    MOPHI_HD bool operator<=(const Real3<Real>& other) const;
    MOPHI_HD bool operator>=(const Real3<Real>& other) const;
    MOPHI_HD bool operator<(const Real3<Real>& other) const;
    MOPHI_HD bool operator>(const Real3<Real>& other) const;
    MOPHI_HD bool operator==(const Real3<Real>& other) const;
    MOPHI_HD bool operator!=(const Real3<Real>& other) const;

    // FUNCTIONS

    /// Set this vector to the sum of A and B: this = A + B
    MOPHI_HD void Add(const Real3<Real>& A, const Real3<Real>& B);

    /// Set this vector to the difference of A and B: this = A - B
    MOPHI_HD void Sub(const Real3<Real>& A, const Real3<Real>& B);

    /// Set this vector to the product of a vector A and scalar s: this = A * s
    MOPHI_HD void Mul(const Real3<Real>& A, Real s);

    /// Scale this vector by a scalar: this *= s
    MOPHI_HD void Scale(Real s);

    /// Set this vector to the cross product of A and B: this = A x B
    MOPHI_HD void Cross(const Real3<Real>& A, const Real3<Real>& B);

    /// Return the cross product with another vector: result = this x other
    MOPHI_HD Real3<Real> Cross(const Real3<Real> other) const;

    /// Return the dot product with another vector: result = this ^ B
    MOPHI_HD Real Dot(const Real3<Real>& B) const;

    /// Normalize this vector in place, so that its euclidean length is 1.
    /// Return false if the original vector had zero length (in which case the vector
    /// is set to [1,0,0]) and return true otherwise.
    MOPHI_HD bool Normalize();

    /// Return a normalized copy of this vector, with euclidean length = 1.
    /// Not to be confused with Normalize() which normalizes in place.
    MOPHI_HD Real3<Real> GetNormalized() const;

    /// Impose a new length to the vector, keeping the direction unchanged.
    MOPHI_HD void SetLength(Real s);

    /// Output three orthonormal vectors considering this vector along X axis.
    /// Optionally, the \a z_sugg vector can be used to suggest the Z axis.
    /// It is recommended to set \a y_sugg to be not parallel to this vector.
    /// The Z axis will be orthogonal to X and \a y_sugg.
    /// Rely on Gram-Schmidt orthonormalization.
    MOPHI_HD void GetDirectionAxesAsX(Real3<Real>& Vx,
                                      Real3<Real>& Vy,
                                      Real3<Real>& Vz,
                                      Real3<Real> y_sugg = Real3<Real>(0, 1, 0)) const;

    /// Output three orthonormal vectors considering this vector along Y axis.
    /// Optionally, the \a z_sugg vector can be used to suggest the Z axis.
    /// It is recommended to set \a z_sugg to be not parallel to this vector.
    /// Rely on Gram-Schmidt orthonormalization.
    MOPHI_HD void GetDirectionAxesAsY(Real3<Real>& Vx,
                                      Real3<Real>& Vy,
                                      Real3<Real>& Vz,
                                      Real3<Real> z_sugg = Real3<Real>(0, 0, 1)) const;

    /// Output three orthonormal vectors considering this vector along Y axis.
    /// Optionally, the \a x_sugg vector can be used to suggest the X axis.
    /// It is recommended to set \a x_sugg to be not parallel to this vector.
    /// Rely on Gram-Schmidt orthonormalization.
    MOPHI_HD void GetDirectionAxesAsZ(Real3<Real>& Vx,
                                      Real3<Real>& Vy,
                                      Real3<Real>& Vz,
                                      Real3<Real> x_sugg = Real3<Real>(1, 0, 0)) const;

    /// Return the index of the largest component in absolute value.
    MOPHI_HD int GetMaxComponent() const;

    /// Return a unit vector orthogonal to this vector
    MOPHI_HD Real3<Real> GetOrthogonalVector() const;

  private:
    Real m_data[3];

    /// Declaration of friend classes
    template <typename RealB>
    friend class Real3;
};

// -----------------------------------------------------------------------------

/// Alias for double-precision vectors.
/// <pre>
/// Instead of writing
///    Real3<double> v;
/// or
///    Real3d v;
/// you can use:
///    Real3d v;
/// </pre>
typedef Real3<double> Real3d;

/// Alias for single-precision vectors.
/// <pre>
/// Instead of writing
///    Real3<float> v;
/// you can use:
///    Real3f v;
/// </pre>
typedef Real3<float> Real3f;

/// Alias for integer vectors.
/// <pre>
/// Instead of writing
///    Real3<int> v;
/// you can use:
///    Real3i v;
/// </pre>
typedef Real3<int> Real3i;

/// Alias for integer vectors.
/// <pre>
/// Instead of writing
///    Real3<long int> v;
/// you can use:
///    Real3l v;
/// </pre>
typedef Real3<long int> Real3l;

/// Alias for bool vectors.
/// <pre>
/// Instead of writing
///    Real3<bool> v;
/// you can use:
///    Real3b v;
/// </pre>
typedef Real3<bool> Real3b;

// -----------------------------------------------------------------------------
// CONSTANTS

const Real3f VECT_NULL_FLOAT(0., 0., 0.);
const Real3f VECT_X_FLOAT(1., 0., 0.);
const Real3f VECT_Y_FLOAT(0., 1., 0.);
const Real3f VECT_Z_FLOAT(0., 0., 1.);

const Real3d VECT_NULL_DOUBLE(0., 0., 0.);
const Real3d VECT_X_DOUBLE(1., 0., 0.);
const Real3d VECT_Y_DOUBLE(0., 1., 0.);
const Real3d VECT_Z_DOUBLE(0., 0., 1.);

// -----------------------------------------------------------------------------
// STATIC VECTOR MATH OPERATIONS

// These functions are here for users who prefer to use global functions instead of Real3 member functions.

template <class RealA, class RealB>
MOPHI_HD RealA Vdot(const Real3<RealA>& va, const Real3<RealB>& vb) {
    return (RealA)((va.x() * vb.x()) + (va.y() * vb.y()) + (va.z() * vb.z()));
}

template <class RealA>
MOPHI_HD void Vset(Real3<RealA>& v, RealA mx, RealA my, RealA mz) {
    v.x() = mx;
    v.y() = my;
    v.z() = mz;
}

template <class RealA, class RealB>
MOPHI_HD Real3<RealA> Vadd(const Real3<RealA>& va, const Real3<RealB>& vb) {
    Real3<RealA> result;
    result.x() = va.x() + vb.x();
    result.y() = va.y() + vb.y();
    result.z() = va.z() + vb.z();
    return result;
}

template <class RealA, class RealB>
MOPHI_HD Real3<RealA> Vsub(const Real3<RealA>& va, const Real3<RealB>& vb) {
    Real3<RealA> result;
    result.x() = va.x() - vb.x();
    result.y() = va.y() - vb.y();
    result.z() = va.z() - vb.z();
    return result;
}

template <class RealA, class RealB>
MOPHI_HD Real3<RealA> Vcross(const Real3<RealA>& va, const Real3<RealB>& vb) {
    Real3<RealA> result;
    result.x() = (va.y() * vb.z()) - (va.z() * vb.y());
    result.y() = (va.z() * vb.x()) - (va.x() * vb.z());
    result.z() = (va.x() * vb.y()) - (va.y() * vb.x());
    return result;
}

template <class RealA, class RealB>
MOPHI_HD Real3<RealA> Vmul(const Real3<RealA>& va, RealB fact) {
    Real3<RealA> result;
    result.x() = va.x() * (RealA)fact;
    result.y() = va.y() * (RealA)fact;
    result.z() = va.z() * (RealA)fact;
    return result;
}

template <class RealA>
MOPHI_HD RealA Vlength(const Real3<RealA>& va) {
    return (RealA)va.Length();
}

template <class RealA>
MOPHI_HD Real3<RealA> Vnorm(const Real3<RealA>& va) {
    Real3<RealA> result(va);
    result.Normalize();
    return result;
}

template <class RealA, class RealB>
MOPHI_HD bool Vequal(const Real3<RealA>& va, const Real3<RealB>& vb) {
    return (va == vb);
}

template <class RealA>
MOPHI_HD bool Vnotnull(const Real3<RealA>& va) {
    return (va.x() != 0 || va.y() != 0 || va.z() != 0);
}

template <class RealA>
MOPHI_HD Real3<RealA> Vmin(const Real3<RealA>& va, const Real3<RealA>& vb) {
    Real3<RealA> result;
    result.x() = std::min(va.x(), vb.x());
    result.y() = std::min(va.y(), vb.y());
    result.z() = std::min(va.z(), vb.z());
    return result;
}

template <class RealA>
MOPHI_HD Real3<RealA> Vmax(const Real3<RealA>& va, const Real3<RealA>& vb) {
    Real3<RealA> result;
    result.x() = std::max(va.x(), vb.x());
    result.y() = std::max(va.y(), vb.y());
    result.z() = std::max(va.z(), vb.z());
    return result;
}

// Gets the angle of the projection on the YZ plane respect to
// the Y vector, as spinning about X.
template <class RealA>
MOPHI_HD double VangleRX(const Real3<RealA>& va) {
    Real3<RealA> vproj;
    vproj.x() = 0;
    vproj.y() = va.y();
    vproj.z() = va.z();
    vproj = Vnorm(vproj);
    if (vproj.x() == 1)
        return 0;
    return acos(vproj.y());
}

// The reverse of the two previous functions, gets the vector
// given the angle above the normal to YZ plane and the angle
// of rotation on X
template <class RealA>
MOPHI_HD Real3<RealA> VfromPolar(double norm_angle, double pol_angle) {
    Real3d res;
    double projlen;
    res.x() = cos(norm_angle);  // 1) rot 'norm.angle'about z
    res.y() = sin(norm_angle);
    res.z() = 0;
    projlen = res.y();
    res.y() = projlen * cos(pol_angle);
    res.z() = projlen * sin(pol_angle);
    return res;
}

/// Insertion of a 3D vector to output stream.
template <typename Real>
inline std::ostream& operator<<(std::ostream& out, const Real3<Real>& v) {
    out << v.x() << "  " << v.y() << "  " << v.z();
    return out;
}

// =============================================================================
// IMPLEMENTATION OF Real3<Real> methods
// =============================================================================

// -----------------------------------------------------------------------------
// Constructors

template <class Real>
inline MOPHI_HD Real3<Real>::Real3() {
    m_data[0] = 0;
    m_data[1] = 0;
    m_data[2] = 0;
}

template <class Real>
inline MOPHI_HD Real3<Real>::Real3(Real a) {
    m_data[0] = a;
    m_data[1] = a;
    m_data[2] = a;
}

template <class Real>
inline MOPHI_HD Real3<Real>::Real3(Real x, Real y, Real z) {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
}

template <class Real>
inline MOPHI_HD Real3<Real>::Real3(const Real3<Real>& other) {
    m_data[0] = other.m_data[0];
    m_data[1] = other.m_data[1];
    m_data[2] = other.m_data[2];
}

template <class Real>
template <class RealB>
inline MOPHI_HD Real3<Real>::Real3(const Real3<RealB>& other) {
    m_data[0] = static_cast<Real>(other.m_data[0]);
    m_data[1] = static_cast<Real>(other.m_data[1]);
    m_data[2] = static_cast<Real>(other.m_data[2]);
}

// -----------------------------------------------------------------------------
// Subscript operators

template <class Real>
inline MOPHI_HD Real& Real3<Real>::operator[](unsigned index) {
    // assert(index < 3);
    return m_data[index];
}

template <class Real>
inline const MOPHI_HD Real& Real3<Real>::operator[](unsigned index) const {
    // assert(index < 3);
    return m_data[index];
}

// -----------------------------------------------------------------------------
// Assignments

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator=(const Real3<Real>& other) {
    if (&other == this)
        return *this;
    m_data[0] = other.m_data[0];
    m_data[1] = other.m_data[1];
    m_data[2] = other.m_data[2];
    return *this;
}

template <class Real>
template <class RealB>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator=(const Real3<RealB>& other) {
    m_data[0] = static_cast<Real>(other.m_data[0]);
    m_data[1] = static_cast<Real>(other.m_data[1]);
    m_data[2] = static_cast<Real>(other.m_data[2]);
    return *this;
}

// -----------------------------------------------------------------------------
// Sign operators

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator+() const {
    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator-() const {
    return Real3<Real>(-m_data[0], -m_data[1], -m_data[2]);
}

// -----------------------------------------------------------------------------
// Arithmetic operations

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator+(const Real3<Real>& other) const {
    Real3<Real> v;

    v.m_data[0] = m_data[0] + other.m_data[0];
    v.m_data[1] = m_data[1] + other.m_data[1];
    v.m_data[2] = m_data[2] + other.m_data[2];

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator-(const Real3<Real>& other) const {
    Real3<Real> v;

    v.m_data[0] = m_data[0] - other.m_data[0];
    v.m_data[1] = m_data[1] - other.m_data[1];
    v.m_data[2] = m_data[2] - other.m_data[2];

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator*(const Real3<Real>& other) const {
    Real3<Real> v;

    v.m_data[0] = m_data[0] * other.m_data[0];
    v.m_data[1] = m_data[1] * other.m_data[1];
    v.m_data[2] = m_data[2] * other.m_data[2];

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator/(const Real3<Real>& other) const {
    Real3<Real> v;

    v.m_data[0] = m_data[0] / other.m_data[0];
    v.m_data[1] = m_data[1] / other.m_data[1];
    v.m_data[2] = m_data[2] / other.m_data[2];

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator*(Real s) const {
    Real3<Real> v;

    v.m_data[0] = m_data[0] * s;
    v.m_data[1] = m_data[1] * s;
    v.m_data[2] = m_data[2] * s;

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::operator/(Real s) const {
    Real oos = 1 / s;
    Real3<Real> v;

    v.m_data[0] = m_data[0] * oos;
    v.m_data[1] = m_data[1] * oos;
    v.m_data[2] = m_data[2] * oos;

    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator+=(const Real3<Real>& other) {
    m_data[0] += other.m_data[0];
    m_data[1] += other.m_data[1];
    m_data[2] += other.m_data[2];

    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator-=(const Real3<Real>& other) {
    m_data[0] -= other.m_data[0];
    m_data[1] -= other.m_data[1];
    m_data[2] -= other.m_data[2];

    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator*=(const Real3<Real>& other) {
    m_data[0] *= other.m_data[0];
    m_data[1] *= other.m_data[1];
    m_data[2] *= other.m_data[2];

    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator/=(const Real3<Real>& other) {
    m_data[0] /= other.m_data[0];
    m_data[1] /= other.m_data[1];
    m_data[2] /= other.m_data[2];

    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator*=(Real s) {
    m_data[0] *= s;
    m_data[1] *= s;
    m_data[2] *= s;

    return *this;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator/=(Real s) {
    Real oos = 1 / s;

    m_data[0] *= oos;
    m_data[1] *= oos;
    m_data[2] *= oos;

    return *this;
}

// -----------------------------------------------------------------------------
// Vector operations

template <class Real>
inline MOPHI_HD Real Real3<Real>::operator^(const Real3<Real>& other) const {
    return this->Dot(other);
}

template <class Real>
MOPHI_HD Real3<Real> Real3<Real>::operator%(const Real3<Real>& other) const {
    Real3<Real> v;
    v.Cross(*this, other);
    return v;
}

template <class Real>
inline MOPHI_HD Real3<Real>& Real3<Real>::operator%=(const Real3<Real>& other) {
    this->Cross(*this, other);
    return *this;
}

// -----------------------------------------------------------------------------
// Comparison operations

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator<=(const Real3<Real>& other) const {
    return m_data[0] <= other.m_data[0] && m_data[1] <= other.m_data[1] && m_data[2] <= other.m_data[2];
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator>=(const Real3<Real>& other) const {
    return m_data[0] >= other.m_data[0] && m_data[1] >= other.m_data[1] && m_data[2] >= other.m_data[2];
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator<(const Real3<Real>& other) const {
    return m_data[0] < other.m_data[0] && m_data[1] < other.m_data[1] && m_data[2] < other.m_data[2];
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator>(const Real3<Real>& other) const {
    return m_data[0] > other.m_data[0] && m_data[1] > other.m_data[1] && m_data[2] > other.m_data[2];
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator==(const Real3<Real>& other) const {
    return other.m_data[0] == m_data[0] && other.m_data[1] == m_data[1] && other.m_data[2] == m_data[2];
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::operator!=(const Real3<Real>& other) const {
    return !(*this == other);
}

// -----------------------------------------------------------------------------
// Functions

template <class Real>
inline MOPHI_HD void Real3<Real>::Set(Real x, Real y, Real z) {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Set(const Real3<Real>& v) {
    m_data[0] = v.m_data[0];
    m_data[1] = v.m_data[1];
    m_data[2] = v.m_data[2];
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Set(Real s) {
    m_data[0] = s;
    m_data[1] = s;
    m_data[2] = s;
}

/// Sets the vector as a null vector
template <class Real>
inline MOPHI_HD void Real3<Real>::SetNull() {
    m_data[0] = 0;
    m_data[1] = 0;
    m_data[2] = 0;
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::IsNull() const {
    return m_data[0] == 0 && m_data[1] == 0 && m_data[2] == 0;
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::Equals(const Real3<Real>& other) const {
    return (other.m_data[0] == m_data[0]) && (other.m_data[1] == m_data[1]) && (other.m_data[2] == m_data[2]);
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::Equals(const Real3<Real>& other, Real tol) const {
    return (std::abs(other.m_data[0] - m_data[0]) < tol) && (std::abs(other.m_data[1] - m_data[1]) < tol) &&
           (std::abs(other.m_data[2] - m_data[2]) < tol);
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Add(const Real3<Real>& A, const Real3<Real>& B) {
    m_data[0] = A.m_data[0] + B.m_data[0];
    m_data[1] = A.m_data[1] + B.m_data[1];
    m_data[2] = A.m_data[2] + B.m_data[2];
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Sub(const Real3<Real>& A, const Real3<Real>& B) {
    m_data[0] = A.m_data[0] - B.m_data[0];
    m_data[1] = A.m_data[1] - B.m_data[1];
    m_data[2] = A.m_data[2] - B.m_data[2];
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Mul(const Real3<Real>& A, Real s) {
    m_data[0] = A.m_data[0] * s;
    m_data[1] = A.m_data[1] * s;
    m_data[2] = A.m_data[2] * s;
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Scale(Real s) {
    m_data[0] *= s;
    m_data[1] *= s;
    m_data[2] *= s;
}

template <class Real>
inline MOPHI_HD void Real3<Real>::Cross(const Real3<Real>& A, const Real3<Real>& B) {
    m_data[0] = (A.m_data[1] * B.m_data[2]) - (A.m_data[2] * B.m_data[1]);
    m_data[1] = (A.m_data[2] * B.m_data[0]) - (A.m_data[0] * B.m_data[2]);
    m_data[2] = (A.m_data[0] * B.m_data[1]) - (A.m_data[1] * B.m_data[0]);
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::Cross(const Real3<Real> other) const {
    Real3<Real> v;
    v.Cross(*this, other);
    return v;
}

template <class Real>
inline MOPHI_HD Real Real3<Real>::Dot(const Real3<Real>& B) const {
    return (m_data[0] * B.m_data[0]) + (m_data[1] * B.m_data[1]) + (m_data[2] * B.m_data[2]);
}

template <class Real>
inline MOPHI_HD Real Real3<Real>::Length() const {
    return sqrt(Length2());
}

template <class Real>
inline MOPHI_HD Real Real3<Real>::Length2() const {
    return this->Dot(*this);
}

template <class Real>
inline MOPHI_HD Real Real3<Real>::LengthInf() const {
    return std::max(std::max(std::abs(m_data[0]), std::abs(m_data[1])), std::abs(m_data[2]));
}

template <class Real>
inline MOPHI_HD bool Real3<Real>::Normalize() {
    Real length = this->Length();
    if (length < std::numeric_limits<Real>::min()) {
        m_data[0] = 1;
        m_data[1] = 0;
        m_data[2] = 0;
        return false;
    }
    this->Scale(1 / length);
    return true;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::GetNormalized() const {
    Real3<Real> v(*this);
    v.Normalize();
    return v;
}

template <class Real>
inline MOPHI_HD void Real3<Real>::SetLength(Real s) {
    Normalize();
    Scale(s);
}

template <class Real>
inline MOPHI_HD void Real3<Real>::GetDirectionAxesAsX(Real3<Real>& Vx,
                                                      Real3<Real>& Vy,
                                                      Real3<Real>& Vz,
                                                      Real3<Real> y_sugg) const {
    Vx = *this;
    bool success = Vx.Normalize();
    if (!success)
        Vx = Real3<Real>(1, 0, 0);

    Vz.Cross(Vx, y_sugg);
    success = Vz.Normalize();
    if (!success) {
        char idx = 0;
        while (!success) {
            y_sugg[idx] += 1.0;
            Vz.Cross(Vx, y_sugg);
            success = Vz.Normalize();
            ++idx;
        }
    }

    Vy.Cross(Vz, Vx);
}

template <class Real>
inline MOPHI_HD void Real3<Real>::GetDirectionAxesAsY(Real3<Real>& Vx,
                                                      Real3<Real>& Vy,
                                                      Real3<Real>& Vz,
                                                      Real3<Real> z_sugg) const {
    Vy = *this;
    bool success = Vy.Normalize();
    if (!success)
        Vy = Real3<Real>(0, 1, 0);

    Vx.Cross(Vy, z_sugg);
    success = Vx.Normalize();
    if (!success) {
        char idx = 0;
        while (!success) {
            z_sugg[idx] += 1.0;
            Vx.Cross(Vy, z_sugg);
            success = Vx.Normalize();
            ++idx;
        }
    }

    Vy.Cross(Vz, Vx);
}

template <class Real>
inline MOPHI_HD void Real3<Real>::GetDirectionAxesAsZ(Real3<Real>& Vx,
                                                      Real3<Real>& Vy,
                                                      Real3<Real>& Vz,
                                                      Real3<Real> x_sugg) const {
    Vz = *this;
    bool success = Vz.Normalize();
    if (!success)
        Vz = Real3<Real>(0, 0, 1);

    Vy.Cross(Vz, x_sugg);
    success = Vy.Normalize();

    if (!success) {
        char idx = 0;
        while (!success) {
            x_sugg[idx] += 1.0;
            Vy.Cross(Vz, x_sugg);
            success = Vy.Normalize();
            ++idx;
        }
    }

    Vx.Cross(Vy, Vz);
}

template <class Real>
inline MOPHI_HD int Real3<Real>::GetMaxComponent() const {
    int idx = 0;
    Real max = std::abs(m_data[0]);
    if (std::abs(m_data[1]) > max) {
        idx = 1;
        max = m_data[1];
    }
    if (std::abs(m_data[2]) > max) {
        idx = 2;
        max = m_data[2];
    }
    return idx;
}

template <class Real>
inline MOPHI_HD Real3<Real> Real3<Real>::GetOrthogonalVector() const {
    int idx1 = this->GetMaxComponent();
    int idx2 = (idx1 + 1) % 3;  // cycle to the next component
    int idx3 = (idx2 + 1) % 3;  // cycle to the next component

    // Construct v2 by rotating in the plane containing the maximum component
    Real3<Real> v2(-m_data[idx2], m_data[idx1], m_data[idx3]);

    // Construct the normal vector
    Real3<Real> ortho = Cross(v2);
    ortho.Normalize();
    return ortho;
}

// -----------------------------------------------------------------------------
// Reversed operators

/// Operator for scaling the vector by a scalar value, as s*V
template <class Real>
MOPHI_HD Real3<Real> operator*(Real s, const Real3<Real>& V) {
    return Real3<Real>(V.x() * s, V.y() * s, V.z() * s);
}

}  // namespace mophi

#endif
