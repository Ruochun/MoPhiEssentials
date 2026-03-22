//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file Real4.hpp
 * @brief 4-component vector class mophi::Real4<Real>, modeled after Real3.
 *
 * Provides the same API as Real3 but for 4-element vectors (x, y, z, w).
 * Utility methods (Clamp, Lerp, Floor, Frac, Fmod, Abs) are included;
 * Reflect is not applicable to 4D vectors and is omitted.
 *
 * Convenience typedefs:
 *   Real4f  = Real4<float>
 *   Real4d  = Real4<double>
 */

#ifndef MOPHI_REAL4_HPP
#define MOPHI_REAL4_HPP

#include <algorithm>
#include <cmath>
#include <ostream>

#ifndef MOPHI_HD
    #ifdef __CUDACC__
        #define MOPHI_HD __host__ __device__
    #else
        #define MOPHI_HD
    #endif
#endif

namespace mophi {

/// 4-component vector templated on scalar type Real (default: float).
template <class Real = float>
class alignas(sizeof(Real) * 4) Real4 {
  public:
    // CONSTRUCTORS
    MOPHI_HD Real4();
    MOPHI_HD Real4(Real x, Real y, Real z, Real w);
    MOPHI_HD Real4(Real a);
    MOPHI_HD Real4(const Real4<Real>& other);

    /// Copy constructor with type change.
    template <class RealB>
    MOPHI_HD Real4(const Real4<RealB>& other);

    // ACCESSORS
    MOPHI_HD Real& x() { return m_data[0]; }
    MOPHI_HD Real& y() { return m_data[1]; }
    MOPHI_HD Real& z() { return m_data[2]; }
    MOPHI_HD Real& w() { return m_data[3]; }
    const MOPHI_HD Real& x() const { return m_data[0]; }
    const MOPHI_HD Real& y() const { return m_data[1]; }
    const MOPHI_HD Real& z() const { return m_data[2]; }
    const MOPHI_HD Real& w() const { return m_data[3]; }

    MOPHI_HD Real* data() { return m_data; }
    const MOPHI_HD Real* data() const { return m_data; }

    // SETTERS
    MOPHI_HD void Set(Real x, Real y, Real z, Real w);
    MOPHI_HD void Set(const Real4<Real>& v);
    MOPHI_HD void Set(Real s);
    MOPHI_HD void SetNull();

    // TESTS
    MOPHI_HD bool IsNull() const;
    MOPHI_HD bool Equals(const Real4<Real>& other) const;
    MOPHI_HD bool Equals(const Real4<Real>& other, Real tol) const;

    // VECTOR NORMS
    /// Euclidean length.
    MOPHI_HD Real Length() const;
    /// Squared euclidean length.
    MOPHI_HD Real Length2() const;
    /// Infinity norm (max absolute value of components).
    MOPHI_HD Real LengthInf() const;

    // OPERATORS
    MOPHI_HD Real& operator[](unsigned index);
    const MOPHI_HD Real& operator[](unsigned index) const;

    MOPHI_HD Real4<Real>& operator=(const Real4<Real>& other);
    template <class RealB>
    MOPHI_HD Real4<Real>& operator=(const Real4<RealB>& other);

    MOPHI_HD Real4<Real> operator+() const;
    MOPHI_HD Real4<Real> operator-() const;

    MOPHI_HD Real4<Real> operator+(const Real4<Real>& other) const;
    MOPHI_HD Real4<Real>& operator+=(const Real4<Real>& other);
    MOPHI_HD Real4<Real> operator-(const Real4<Real>& other) const;
    MOPHI_HD Real4<Real>& operator-=(const Real4<Real>& other);
    MOPHI_HD Real4<Real> operator*(const Real4<Real>& other) const;
    MOPHI_HD Real4<Real>& operator*=(const Real4<Real>& other);
    MOPHI_HD Real4<Real> operator/(const Real4<Real>& other) const;
    MOPHI_HD Real4<Real>& operator/=(const Real4<Real>& other);
    MOPHI_HD Real4<Real> operator*(Real s) const;
    MOPHI_HD Real4<Real>& operator*=(Real s);
    MOPHI_HD Real4<Real> operator/(Real v) const;
    MOPHI_HD Real4<Real>& operator/=(Real v);

    MOPHI_HD bool operator==(const Real4<Real>& other) const;
    MOPHI_HD bool operator!=(const Real4<Real>& other) const;

    // DOT PRODUCT (^ operator)
    MOPHI_HD Real operator^(const Real4<Real>& other) const;

    // MATH FUNCTIONS
    MOPHI_HD Real Dot(const Real4<Real>& B) const;
    MOPHI_HD void Add(const Real4<Real>& A, const Real4<Real>& B);
    MOPHI_HD void Sub(const Real4<Real>& A, const Real4<Real>& B);
    MOPHI_HD void Mul(const Real4<Real>& A, Real s);
    MOPHI_HD void Scale(Real s);
    MOPHI_HD bool Normalize();
    MOPHI_HD Real4<Real> GetNormalized() const;

    // UTILITY METHODS

    /// Return a new vector with each component clamped to [lo, hi].
    MOPHI_HD Real4<Real> Clamp(Real lo, Real hi) const;

    /// Return a new vector with each component clamped element-wise to [lo, hi].
    MOPHI_HD Real4<Real> Clamp(const Real4<Real>& lo, const Real4<Real>& hi) const;

    /// Linear interpolation between this and other by t in [0,1].
    MOPHI_HD Real4<Real> Lerp(const Real4<Real>& other, Real t) const;

    /// Element-wise floor.
    MOPHI_HD Real4<Real> Floor() const;

    /// Element-wise fractional part.
    MOPHI_HD Real4<Real> Frac() const;

    /// Element-wise fmod.
    MOPHI_HD Real4<Real> Fmod(const Real4<Real>& divisor) const;

    /// Element-wise absolute value.
    MOPHI_HD Real4<Real> Abs() const;

  private:
    Real m_data[4];

    template <typename RealB>
    friend class Real4;
};

// -----------------------------------------------------------------------------
// Typedefs

/// Double-precision 4-vector.
typedef Real4<double> Real4d;
/// Single-precision 4-vector.
typedef Real4<float> Real4f;

// -----------------------------------------------------------------------------
// Stream output

template <typename Real>
inline std::ostream& operator<<(std::ostream& out, const Real4<Real>& v) {
    out << v.x() << "  " << v.y() << "  " << v.z() << "  " << v.w();
    return out;
}

// =============================================================================
// IMPLEMENTATION
// =============================================================================

// Constructors

template <class Real>
inline MOPHI_HD Real4<Real>::Real4() {
    m_data[0] = m_data[1] = m_data[2] = m_data[3] = 0;
}

template <class Real>
inline MOPHI_HD Real4<Real>::Real4(Real a) {
    m_data[0] = m_data[1] = m_data[2] = m_data[3] = a;
}

template <class Real>
inline MOPHI_HD Real4<Real>::Real4(Real x, Real y, Real z, Real w) {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
}

template <class Real>
inline MOPHI_HD Real4<Real>::Real4(const Real4<Real>& other) {
    m_data[0] = other.m_data[0];
    m_data[1] = other.m_data[1];
    m_data[2] = other.m_data[2];
    m_data[3] = other.m_data[3];
}

template <class Real>
template <class RealB>
inline MOPHI_HD Real4<Real>::Real4(const Real4<RealB>& other) {
    m_data[0] = static_cast<Real>(other.m_data[0]);
    m_data[1] = static_cast<Real>(other.m_data[1]);
    m_data[2] = static_cast<Real>(other.m_data[2]);
    m_data[3] = static_cast<Real>(other.m_data[3]);
}

// Subscript

template <class Real>
inline MOPHI_HD Real& Real4<Real>::operator[](unsigned index) {
    return m_data[index];
}
template <class Real>
inline const MOPHI_HD Real& Real4<Real>::operator[](unsigned index) const {
    return m_data[index];
}

// Assignments

template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator=(const Real4<Real>& other) {
    if (&other == this) return *this;
    m_data[0] = other.m_data[0];
    m_data[1] = other.m_data[1];
    m_data[2] = other.m_data[2];
    m_data[3] = other.m_data[3];
    return *this;
}
template <class Real>
template <class RealB>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator=(const Real4<RealB>& other) {
    m_data[0] = static_cast<Real>(other.m_data[0]);
    m_data[1] = static_cast<Real>(other.m_data[1]);
    m_data[2] = static_cast<Real>(other.m_data[2]);
    m_data[3] = static_cast<Real>(other.m_data[3]);
    return *this;
}

// Sign operators

template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator+() const { return *this; }
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator-() const {
    return Real4<Real>(-m_data[0], -m_data[1], -m_data[2], -m_data[3]);
}

// Arithmetic

template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator+(const Real4<Real>& o) const {
    return Real4<Real>(m_data[0]+o.m_data[0], m_data[1]+o.m_data[1],
                       m_data[2]+o.m_data[2], m_data[3]+o.m_data[3]);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator+=(const Real4<Real>& o) {
    m_data[0]+=o.m_data[0]; m_data[1]+=o.m_data[1];
    m_data[2]+=o.m_data[2]; m_data[3]+=o.m_data[3];
    return *this;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator-(const Real4<Real>& o) const {
    return Real4<Real>(m_data[0]-o.m_data[0], m_data[1]-o.m_data[1],
                       m_data[2]-o.m_data[2], m_data[3]-o.m_data[3]);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator-=(const Real4<Real>& o) {
    m_data[0]-=o.m_data[0]; m_data[1]-=o.m_data[1];
    m_data[2]-=o.m_data[2]; m_data[3]-=o.m_data[3];
    return *this;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator*(const Real4<Real>& o) const {
    return Real4<Real>(m_data[0]*o.m_data[0], m_data[1]*o.m_data[1],
                       m_data[2]*o.m_data[2], m_data[3]*o.m_data[3]);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator*=(const Real4<Real>& o) {
    m_data[0]*=o.m_data[0]; m_data[1]*=o.m_data[1];
    m_data[2]*=o.m_data[2]; m_data[3]*=o.m_data[3];
    return *this;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator/(const Real4<Real>& o) const {
    return Real4<Real>(m_data[0]/o.m_data[0], m_data[1]/o.m_data[1],
                       m_data[2]/o.m_data[2], m_data[3]/o.m_data[3]);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator/=(const Real4<Real>& o) {
    m_data[0]/=o.m_data[0]; m_data[1]/=o.m_data[1];
    m_data[2]/=o.m_data[2]; m_data[3]/=o.m_data[3];
    return *this;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator*(Real s) const {
    return Real4<Real>(m_data[0]*s, m_data[1]*s, m_data[2]*s, m_data[3]*s);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator*=(Real s) {
    m_data[0]*=s; m_data[1]*=s; m_data[2]*=s; m_data[3]*=s;
    return *this;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::operator/(Real v) const {
    Real oos = Real(1) / v;
    return Real4<Real>(m_data[0]*oos, m_data[1]*oos, m_data[2]*oos, m_data[3]*oos);
}
template <class Real>
inline MOPHI_HD Real4<Real>& Real4<Real>::operator/=(Real v) {
    Real oos = Real(1) / v;
    m_data[0]*=oos; m_data[1]*=oos; m_data[2]*=oos; m_data[3]*=oos;
    return *this;
}

// Comparison

template <class Real>
inline MOPHI_HD bool Real4<Real>::operator==(const Real4<Real>& o) const {
    return m_data[0]==o.m_data[0] && m_data[1]==o.m_data[1] &&
           m_data[2]==o.m_data[2] && m_data[3]==o.m_data[3];
}
template <class Real>
inline MOPHI_HD bool Real4<Real>::operator!=(const Real4<Real>& o) const {
    return !(*this == o);
}

// Dot product

template <class Real>
inline MOPHI_HD Real Real4<Real>::operator^(const Real4<Real>& other) const {
    return Dot(other);
}
template <class Real>
inline MOPHI_HD Real Real4<Real>::Dot(const Real4<Real>& B) const {
    return m_data[0]*B.m_data[0] + m_data[1]*B.m_data[1] +
           m_data[2]*B.m_data[2] + m_data[3]*B.m_data[3];
}

// Setters

template <class Real>
inline MOPHI_HD void Real4<Real>::Set(Real x, Real y, Real z, Real w) {
    m_data[0]=x; m_data[1]=y; m_data[2]=z; m_data[3]=w;
}
template <class Real>
inline MOPHI_HD void Real4<Real>::Set(const Real4<Real>& v) {
    m_data[0]=v.m_data[0]; m_data[1]=v.m_data[1];
    m_data[2]=v.m_data[2]; m_data[3]=v.m_data[3];
}
template <class Real>
inline MOPHI_HD void Real4<Real>::Set(Real s) {
    m_data[0]=m_data[1]=m_data[2]=m_data[3]=s;
}
template <class Real>
inline MOPHI_HD void Real4<Real>::SetNull() {
    m_data[0]=m_data[1]=m_data[2]=m_data[3]=0;
}

// Tests

template <class Real>
inline MOPHI_HD bool Real4<Real>::IsNull() const {
    return m_data[0]==0 && m_data[1]==0 && m_data[2]==0 && m_data[3]==0;
}
template <class Real>
inline MOPHI_HD bool Real4<Real>::Equals(const Real4<Real>& o) const {
    return m_data[0]==o.m_data[0] && m_data[1]==o.m_data[1] &&
           m_data[2]==o.m_data[2] && m_data[3]==o.m_data[3];
}
template <class Real>
inline MOPHI_HD bool Real4<Real>::Equals(const Real4<Real>& o, Real tol) const {
    return std::abs(m_data[0]-o.m_data[0]) < tol && std::abs(m_data[1]-o.m_data[1]) < tol &&
           std::abs(m_data[2]-o.m_data[2]) < tol && std::abs(m_data[3]-o.m_data[3]) < tol;
}

// Norms

template <class Real>
inline MOPHI_HD Real Real4<Real>::Length2() const { return Dot(*this); }

template <class Real>
inline MOPHI_HD Real Real4<Real>::Length() const { return std::sqrt(Length2()); }

template <class Real>
inline MOPHI_HD Real Real4<Real>::LengthInf() const {
    return std::max({std::abs(m_data[0]), std::abs(m_data[1]),
                     std::abs(m_data[2]), std::abs(m_data[3])});
}

// Math functions

template <class Real>
inline MOPHI_HD void Real4<Real>::Add(const Real4<Real>& A, const Real4<Real>& B) {
    m_data[0]=A.m_data[0]+B.m_data[0]; m_data[1]=A.m_data[1]+B.m_data[1];
    m_data[2]=A.m_data[2]+B.m_data[2]; m_data[3]=A.m_data[3]+B.m_data[3];
}
template <class Real>
inline MOPHI_HD void Real4<Real>::Sub(const Real4<Real>& A, const Real4<Real>& B) {
    m_data[0]=A.m_data[0]-B.m_data[0]; m_data[1]=A.m_data[1]-B.m_data[1];
    m_data[2]=A.m_data[2]-B.m_data[2]; m_data[3]=A.m_data[3]-B.m_data[3];
}
template <class Real>
inline MOPHI_HD void Real4<Real>::Mul(const Real4<Real>& A, Real s) {
    m_data[0]=A.m_data[0]*s; m_data[1]=A.m_data[1]*s;
    m_data[2]=A.m_data[2]*s; m_data[3]=A.m_data[3]*s;
}
template <class Real>
inline MOPHI_HD void Real4<Real>::Scale(Real s) {
    m_data[0]*=s; m_data[1]*=s; m_data[2]*=s; m_data[3]*=s;
}
template <class Real>
inline MOPHI_HD bool Real4<Real>::Normalize() {
    Real len = Length();
    if (len < Real(1.17549435e-38f)) {
        m_data[0] = 1; m_data[1] = 0; m_data[2] = 0; m_data[3] = 0;
        return false;
    }
    Scale(Real(1) / len);
    return true;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::GetNormalized() const {
    Real4<Real> v(*this);
    v.Normalize();
    return v;
}

// Utility methods

template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Clamp(Real lo, Real hi) const {
    return Real4<Real>(std::min(hi, std::max(lo, m_data[0])),
                       std::min(hi, std::max(lo, m_data[1])),
                       std::min(hi, std::max(lo, m_data[2])),
                       std::min(hi, std::max(lo, m_data[3])));
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Clamp(const Real4<Real>& lo, const Real4<Real>& hi) const {
    return Real4<Real>(std::min(hi.m_data[0], std::max(lo.m_data[0], m_data[0])),
                       std::min(hi.m_data[1], std::max(lo.m_data[1], m_data[1])),
                       std::min(hi.m_data[2], std::max(lo.m_data[2], m_data[2])),
                       std::min(hi.m_data[3], std::max(lo.m_data[3], m_data[3])));
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Lerp(const Real4<Real>& other, Real t) const {
    return (*this) + (other - (*this)) * t;
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Floor() const {
    return Real4<Real>(std::floor(m_data[0]), std::floor(m_data[1]),
                       std::floor(m_data[2]), std::floor(m_data[3]));
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Frac() const {
    return (*this) - Floor();
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Fmod(const Real4<Real>& d) const {
    return Real4<Real>(std::fmod(m_data[0], d.m_data[0]), std::fmod(m_data[1], d.m_data[1]),
                       std::fmod(m_data[2], d.m_data[2]), std::fmod(m_data[3], d.m_data[3]));
}
template <class Real>
inline MOPHI_HD Real4<Real> Real4<Real>::Abs() const {
    return Real4<Real>(std::abs(m_data[0]), std::abs(m_data[1]),
                       std::abs(m_data[2]), std::abs(m_data[3]));
}

// -----------------------------------------------------------------------------
// Reversed operator s*V

template <class Real>
MOPHI_HD Real4<Real> operator*(Real s, const Real4<Real>& V) {
    return Real4<Real>(V.x()*s, V.y()*s, V.z()*s, V.w()*s);
}

}  // namespace mophi

#endif  // MOPHI_REAL4_HPP
