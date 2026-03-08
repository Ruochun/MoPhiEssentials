//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_QUATERNION_H
#define MOPHI_QUATERNION_H

#include "Real3.hpp"
#include <cmath>
#include <iostream>

#ifdef MOPHI_USE_CUDA
    #include <cuda_runtime_api.h>
#endif

#ifndef MOPHI_HD
    #ifdef __CUDACC__
        #define MOPHI_HD __host__ __device__
    #else
        #define MOPHI_HD
    #endif
#endif

namespace mophi {

/// Quaternion class representing 3D rotations.
/// Convention: q = w + xi + yj + zk (scalar-first storage).
/// Designed for dual CPU/GPU use via MOPHI_HD macro.
template <class Real = float>
class Quaternion {
  private:
    Real w_, x_, y_, z_;

  public:
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Default constructor: identity quaternion (no rotation)
    MOPHI_HD Quaternion() : w_(1), x_(0), y_(0), z_(0) {}

    /// Construct from components (w, x, y, z)
    MOPHI_HD Quaternion(Real w, Real x, Real y, Real z) : w_(w), x_(x), y_(y), z_(z) {}

    /// Copy constructor
    MOPHI_HD Quaternion(const Quaternion& other) : w_(other.w_), x_(other.x_), y_(other.y_), z_(other.z_) {}

    // =========================================================================
    // Component access
    // =========================================================================

    MOPHI_HD Real w() const { return w_; }
    MOPHI_HD Real x() const { return x_; }
    MOPHI_HD Real y() const { return y_; }
    MOPHI_HD Real z() const { return z_; }

    MOPHI_HD Real& w() { return w_; }
    MOPHI_HD Real& x() { return x_; }
    MOPHI_HD Real& y() { return y_; }
    MOPHI_HD Real& z() { return z_; }

    /// Set all components
    MOPHI_HD void Set(Real w, Real x, Real y, Real z) {
        w_ = w;
        x_ = x;
        y_ = y;
        z_ = z;
    }

    /// Set to identity quaternion
    MOPHI_HD void SetIdentity() {
        w_ = 1;
        x_ = 0;
        y_ = 0;
        z_ = 0;
    }

    // =========================================================================
    // Norm and normalization
    // =========================================================================

    /// Squared norm (w^2 + x^2 + y^2 + z^2)
    MOPHI_HD Real Norm2() const { return w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_; }

    /// Norm (magnitude)
    MOPHI_HD Real Norm() const { return std::sqrt(Norm2()); }

    /// Normalize in place, returns the original norm
    MOPHI_HD Real Normalize() {
        Real n = Norm();
        if (n > static_cast<Real>(1e-12)) {
            Real inv = static_cast<Real>(1) / n;
            w_ *= inv;
            x_ *= inv;
            y_ *= inv;
            z_ *= inv;
        }
        return n;
    }

    /// Return a normalized copy
    MOPHI_HD Quaternion Normalized() const {
        Quaternion q(*this);
        q.Normalize();
        return q;
    }

    // =========================================================================
    // Conjugate and inverse
    // =========================================================================

    /// Conjugate: q* = (w, -x, -y, -z)
    MOPHI_HD Quaternion Conjugate() const { return Quaternion(w_, -x_, -y_, -z_); }

    /// Inverse (for unit quaternions, same as conjugate)
    MOPHI_HD Quaternion Inverse() const {
        Real n2 = Norm2();
        if (n2 > static_cast<Real>(1e-12)) {
            Real inv = static_cast<Real>(1) / n2;
            return Quaternion(w_ * inv, -x_ * inv, -y_ * inv, -z_ * inv);
        }
        return Quaternion();  // identity as fallback
    }

    // =========================================================================
    // Arithmetic operators
    // =========================================================================

    /// Quaternion multiplication (Hamilton product)
    MOPHI_HD Quaternion operator*(const Quaternion& r) const {
        return Quaternion(w_ * r.w_ - x_ * r.x_ - y_ * r.y_ - z_ * r.z_, w_ * r.x_ + x_ * r.w_ + y_ * r.z_ - z_ * r.y_,
                          w_ * r.y_ - x_ * r.z_ + y_ * r.w_ + z_ * r.x_, w_ * r.z_ + x_ * r.y_ - y_ * r.x_ + z_ * r.w_);
    }

    /// Quaternion addition
    MOPHI_HD Quaternion operator+(const Quaternion& r) const {
        return Quaternion(w_ + r.w_, x_ + r.x_, y_ + r.y_, z_ + r.z_);
    }

    /// Quaternion subtraction
    MOPHI_HD Quaternion operator-(const Quaternion& r) const {
        return Quaternion(w_ - r.w_, x_ - r.x_, y_ - r.y_, z_ - r.z_);
    }

    /// Scalar multiplication
    MOPHI_HD Quaternion operator*(Real s) const { return Quaternion(w_ * s, x_ * s, y_ * s, z_ * s); }

    /// Compound assignment operators
    MOPHI_HD Quaternion& operator*=(const Quaternion& r) {
        *this = *this * r;
        return *this;
    }

    MOPHI_HD Quaternion& operator+=(const Quaternion& r) {
        w_ += r.w_;
        x_ += r.x_;
        y_ += r.y_;
        z_ += r.z_;
        return *this;
    }

    // =========================================================================
    // Rotation operations
    // =========================================================================

    /// Rotate a 3D vector by this quaternion: v' = q * v * q^(-1)
    MOPHI_HD mophi::Real3<Real> Rotate(const mophi::Real3<Real>& v) const {
        // Optimized rotation using the formula:
        // v' = v + 2*w*(u x v) + 2*(u x (u x v))
        // where u = (x, y, z) is the vector part of the quaternion
        mophi::Real3<Real> u(x_, y_, z_);
        mophi::Real3<Real> uv = u.Cross(v);
        mophi::Real3<Real> uuv = u.Cross(uv);
        return v + uv * (static_cast<Real>(2) * w_) + uuv * static_cast<Real>(2);
    }

    /// Rotate a 3D vector by the inverse of this quaternion
    MOPHI_HD mophi::Real3<Real> RotateInverse(const mophi::Real3<Real>& v) const { return Conjugate().Rotate(v); }

    // =========================================================================
    // Conversion: axis-angle
    // =========================================================================

    /// Create quaternion from axis-angle representation
    /// @param axis Unit axis of rotation
    /// @param angle Angle of rotation in radians
    MOPHI_HD static Quaternion FromAxisAngle(const mophi::Real3<Real>& axis, Real angle) {
        Real halfAngle = angle * static_cast<Real>(0.5);
        Real s = std::sin(halfAngle);
        return Quaternion(std::cos(halfAngle), axis.x() * s, axis.y() * s, axis.z() * s);
    }

    /// Convert to axis-angle representation
    /// @param[out] axis Unit axis of rotation
    /// @param[out] angle Angle of rotation in radians
    MOPHI_HD void ToAxisAngle(mophi::Real3<Real>& axis, Real& angle) const {
        Quaternion q = Normalized();
        // Clamp w to [-1, 1] for numerical safety
        Real cw = q.w_;
        if (cw > static_cast<Real>(1))
            cw = static_cast<Real>(1);
        if (cw < static_cast<Real>(-1))
            cw = static_cast<Real>(-1);
        angle = static_cast<Real>(2) * std::acos(cw);
        Real s = std::sqrt(static_cast<Real>(1) - cw * cw);
        if (s < static_cast<Real>(1e-6)) {
            axis.Set(1, 0, 0);  // arbitrary axis for zero rotation
        } else {
            axis.Set(q.x_ / s, q.y_ / s, q.z_ / s);
        }
    }

    // =========================================================================
    // Conversion: Euler angles (ZYX convention)
    // =========================================================================

    /// Create quaternion from Euler angles (roll, pitch, yaw in radians)
    MOPHI_HD static Quaternion FromEulerZYX(Real roll, Real pitch, Real yaw) {
        Real cr = std::cos(roll * static_cast<Real>(0.5));
        Real sr = std::sin(roll * static_cast<Real>(0.5));
        Real cp = std::cos(pitch * static_cast<Real>(0.5));
        Real sp = std::sin(pitch * static_cast<Real>(0.5));
        Real cy = std::cos(yaw * static_cast<Real>(0.5));
        Real sy = std::sin(yaw * static_cast<Real>(0.5));

        return Quaternion(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                          cr * cp * sy - sr * sp * cy);
    }

    // =========================================================================
    // Interpolation
    // =========================================================================

    /// Spherical linear interpolation (SLERP) between two quaternions
    MOPHI_HD static Quaternion Slerp(const Quaternion& a, const Quaternion& b, Real t) {
        Real dot = a.w_ * b.w_ + a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
        Quaternion b2 = b;
        if (dot < 0) {
            b2 = Quaternion(-b.w_, -b.x_, -b.y_, -b.z_);
            dot = -dot;
        }
        if (dot > static_cast<Real>(0.9995)) {
            // Linear interpolation for very close quaternions
            Quaternion result = a * (static_cast<Real>(1) - t) + b2 * t;
            result.Normalize();
            return result;
        }
        Real theta = std::acos(dot);
        Real sinTheta = std::sin(theta);
        Real wa = std::sin((static_cast<Real>(1) - t) * theta) / sinTheta;
        Real wb = std::sin(t * theta) / sinTheta;
        return a * wa + b2 * wb;
    }

    // =========================================================================
    // Stream output
    // =========================================================================

    friend std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
        os << "Quat(" << q.w_ << ", " << q.x_ << ", " << q.y_ << ", " << q.z_ << ")";
        return os;
    }
};

// Type aliases
using Quatf = Quaternion<float>;
using Quatd = Quaternion<double>;

// Scalar * Quaternion
template <class Real>
MOPHI_HD Quaternion<Real> operator*(Real s, const Quaternion<Real>& q) {
    return q * s;
}

}  // namespace mophi

#endif
