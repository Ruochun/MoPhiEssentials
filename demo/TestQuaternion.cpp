//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cmath>
#include <iostream>
#include <core/Quaternion.hpp>

// ---------------------------------------------------------------------------
// Helper: check two doubles are within tolerance
// ---------------------------------------------------------------------------
static void check_near(double a, double b, double tol, const char* msg) {
    double diff = std::abs(a - b);
    if (diff > tol) {
        std::cerr << "FAIL [" << msg << "]: " << a << " vs " << b << " (diff=" << diff << ")" << std::endl;
        assert(false);
    }
}

static void check_vec_near(const mophi::Real3d& a, const mophi::Real3d& b, double tol, const char* msg) {
    check_near(a.x(), b.x(), tol, msg);
    check_near(a.y(), b.y(), tol, msg);
    check_near(a.z(), b.z(), tol, msg);
}

int main() {
    using namespace mophi;
    const double tol = 1e-10;
    const double pi = std::acos(-1.0);

    std::cout << "=== Testing Quaternion Implementation ===" << std::endl;

    // -----------------------------------------------------------------------
    // 1. Default constructor: identity quaternion
    // -----------------------------------------------------------------------
    std::cout << "\n[1] Identity quaternion" << std::endl;
    {
        Quatd q;
        check_near(q.w(), 1.0, tol, "identity w");
        check_near(q.x(), 0.0, tol, "identity x");
        check_near(q.y(), 0.0, tol, "identity y");
        check_near(q.z(), 0.0, tol, "identity z");
        check_near(q.Norm(), 1.0, tol, "identity norm");

        // Rotating any vector by identity should leave it unchanged
        Real3d v(3.0, 1.0, -2.0);
        check_vec_near(q.Rotate(v), v, tol, "identity rotation");
        std::cout << "    ✓ Identity quaternion is correct" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. FromAxisAngle and vector rotation
    // -----------------------------------------------------------------------
    std::cout << "\n[2] Axis-angle construction and rotation" << std::endl;
    {
        // 90° rotation around Z axis: (1,0,0) -> (0,1,0)
        Real3d zAxis(0, 0, 1);
        Quatd q = Quatd::FromAxisAngle(zAxis, pi / 2.0);
        check_near(q.Norm(), 1.0, tol, "axis-angle unit norm");

        Real3d v(1, 0, 0);
        Real3d vr = q.Rotate(v);
        check_vec_near(vr, Real3d(0, 1, 0), 1e-10, "90deg-Z rotation of (1,0,0)");
        std::cout << "    ✓ 90° Z-rotation: (1,0,0) -> " << vr << std::endl;

        // 90° rotation around X axis: (0,1,0) -> (0,0,1)
        Real3d xAxis(1, 0, 0);
        Quatd qx = Quatd::FromAxisAngle(xAxis, pi / 2.0);
        Real3d vy(0, 1, 0);
        Real3d vry = qx.Rotate(vy);
        check_vec_near(vry, Real3d(0, 0, 1), 1e-10, "90deg-X rotation of (0,1,0)");
        std::cout << "    ✓ 90° X-rotation: (0,1,0) -> " << vry << std::endl;

        // 180° rotation around Y axis: (1,0,0) -> (-1,0,0)
        Real3d yAxis(0, 1, 0);
        Quatd qy = Quatd::FromAxisAngle(yAxis, pi);
        Real3d vx(1, 0, 0);
        Real3d vrx = qy.Rotate(vx);
        check_vec_near(vrx, Real3d(-1, 0, 0), 1e-10, "180deg-Y rotation of (1,0,0)");
        std::cout << "    ✓ 180° Y-rotation: (1,0,0) -> " << vrx << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Inverse / conjugate rotation
    // -----------------------------------------------------------------------
    std::cout << "\n[3] Inverse rotation (round-trip)" << std::endl;
    {
        Real3d axis(1, 1, 1);
        axis.Normalize();
        Quatd q = Quatd::FromAxisAngle(axis, pi / 3.0);

        Real3d v(2, -1, 4);
        Real3d vr = q.Rotate(v);
        Real3d vrr = q.RotateInverse(vr);
        check_vec_near(vrr, v, 1e-10, "inverse rotation round-trip");
        std::cout << "    ✓ Round-trip rotation matches original vector" << std::endl;

        // q * q^{-1} should be identity
        Quatd qi = q.Inverse();
        Quatd identity = q * qi;
        check_near(identity.w(), 1.0, tol, "q*q^-1 w");
        check_near(identity.x(), 0.0, tol, "q*q^-1 x");
        check_near(identity.y(), 0.0, tol, "q*q^-1 y");
        check_near(identity.z(), 0.0, tol, "q*q^-1 z");
        std::cout << "    ✓ q * q^(-1) = identity" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4. Quaternion composition
    // -----------------------------------------------------------------------
    std::cout << "\n[4] Quaternion composition" << std::endl;
    {
        // Two 90° rotations around Z should equal one 180° rotation
        Real3d zAxis(0, 0, 1);
        Quatd q90 = Quatd::FromAxisAngle(zAxis, pi / 2.0);
        Quatd q180 = Quatd::FromAxisAngle(zAxis, pi);
        Quatd composed = q90 * q90;

        // Both should rotate (1,0,0) to (-1,0,0)
        Real3d v(1, 0, 0);
        Real3d r1 = q180.Rotate(v);
        Real3d r2 = composed.Rotate(v);
        check_vec_near(r1, r2, 1e-10, "composition equals combined rotation");
        std::cout << "    ✓ q90 * q90 == q180" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 5. Normalization
    // -----------------------------------------------------------------------
    std::cout << "\n[5] Normalization" << std::endl;
    {
        Quatd q(2, 1, -1, 3);
        check_near(q.Norm(), std::sqrt(4 + 1 + 1 + 9), tol, "unnormalized norm");

        Quatd qn = q.Normalized();
        check_near(qn.Norm(), 1.0, tol, "normalized norm");

        // In-place
        Quatd q2(2, 1, -1, 3);
        q2.Normalize();
        check_near(q2.Norm(), 1.0, tol, "in-place normalize norm");
        std::cout << "    ✓ Normalization is correct" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 6. ToAxisAngle round-trip
    // -----------------------------------------------------------------------
    std::cout << "\n[6] ToAxisAngle round-trip" << std::endl;
    {
        Real3d axis0(0, 1, 0);
        double angle0 = pi / 4.0;
        Quatd q = Quatd::FromAxisAngle(axis0, angle0);

        Real3d axisOut;
        double angleOut;
        q.ToAxisAngle(axisOut, angleOut);

        check_near(angleOut, angle0, 1e-10, "ToAxisAngle angle");
        check_vec_near(axisOut, axis0, 1e-10, "ToAxisAngle axis");
        std::cout << "    ✓ ToAxisAngle round-trip is correct" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 7. FromEulerZYX
    // -----------------------------------------------------------------------
    std::cout << "\n[7] FromEulerZYX" << std::endl;
    {
        // Pure yaw (90° around Z): same as FromAxisAngle(Z, pi/2)
        Quatd qEuler = Quatd::FromEulerZYX(0.0, 0.0, pi / 2.0);
        Quatd qAA = Quatd::FromAxisAngle(Real3d(0, 0, 1), pi / 2.0);

        Real3d v(1, 0, 0);
        Real3d r1 = qEuler.Rotate(v);
        Real3d r2 = qAA.Rotate(v);
        check_vec_near(r1, r2, 1e-10, "FromEulerZYX yaw matches axis-angle");
        std::cout << "    ✓ Pure yaw via Euler matches axis-angle result" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 8. Slerp interpolation
    // -----------------------------------------------------------------------
    std::cout << "\n[8] Slerp interpolation" << std::endl;
    {
        Real3d zAxis(0, 0, 1);
        Quatd q0 = Quatd::FromAxisAngle(zAxis, 0.0);
        Quatd q1 = Quatd::FromAxisAngle(zAxis, pi / 2.0);

        // t=0 should give q0, t=1 should give q1, t=0.5 should give 45°
        Quatd qHalf = Quatd::Slerp(q0, q1, 0.5);
        Quatd q45 = Quatd::FromAxisAngle(zAxis, pi / 4.0);

        Real3d v(1, 0, 0);
        Real3d r1 = qHalf.Rotate(v);
        Real3d r2 = q45.Rotate(v);
        check_vec_near(r1, r2, 1e-10, "Slerp t=0.5 matches 45deg rotation");
        std::cout << "    ✓ Slerp(q0, q90, 0.5) == q45" << std::endl;

        // Boundary conditions
        Quatd qAtZero = Quatd::Slerp(q0, q1, 0.0);
        Quatd qAtOne = Quatd::Slerp(q0, q1, 1.0);
        check_vec_near(qAtZero.Rotate(v), q0.Rotate(v), 1e-10, "Slerp t=0");
        check_vec_near(qAtOne.Rotate(v), q1.Rotate(v), 1e-10, "Slerp t=1");
        std::cout << "    ✓ Slerp boundary conditions (t=0 and t=1) are correct" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 9. Scalar multiplication and arithmetic
    // -----------------------------------------------------------------------
    std::cout << "\n[9] Arithmetic operators" << std::endl;
    {
        Quatd a(1, 0, 0, 0);
        Quatd b(0, 1, 0, 0);
        Quatd sum = a + b;
        check_near(sum.w(), 1.0, tol, "sum w");
        check_near(sum.x(), 1.0, tol, "sum x");

        Quatd scaled = a * 3.0;
        check_near(scaled.w(), 3.0, tol, "scalar mul w");

        Quatd scaled2 = 3.0 * a;
        check_near(scaled2.w(), 3.0, tol, "scalar mul (lhs) w");
        std::cout << "    ✓ Arithmetic operators are correct" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== All Quaternion Tests Passed! ===" << std::endl;
    return 0;
}
