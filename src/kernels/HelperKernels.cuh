// DEM device-side helper kernel collection

#ifndef MOPHI_HELPER_KERNELS_CUH
#define MOPHI_HELPER_KERNELS_CUH

#include <common/Defines.hpp>
#include <core/Real3.hpp>

#include <cstdint>
#include <type_traits>
#include <limits>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// A few helper functions
////////////////////////////////////////////////////////////////////////////////

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
    if (b < 0)  // you can check for b == 0 separately and do what you want
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
MOPHI_HD inline T tet_volume(const mophi::Real3<T>& a,
                             const mophi::Real3<T>& b,
                             const mophi::Real3<T>& c,
                             const mophi::Real3<T>& d) {
    return std::abs((b - a) ^ ((c - a) % (d - a))) / 6.0;  // ^ is dot, % is cross
}
template <typename T>
MOPHI_HD inline T tri_area(const mophi::Real3<T>& a, const mophi::Real3<T>& b, const mophi::Real3<T>& c) {
    return 0.5 * ((b - a) % (c - a)).Length();
}
template <typename T>
MOPHI_HD inline void tet_gradN_phys(const mophi::Real3<T>& a,
                                    const mophi::Real3<T>& b,
                                    const mophi::Real3<T>& c,
                                    const mophi::Real3<T>& d,
                                    mophi::Real3<T> g[4]) {
    // ∇N1 = (e2 × e3)/det, ∇N2 = (e3 × e1)/det, ∇N3 = (e1 × e2)/det, ∇N0 = -Σ
    const mophi::Real3<T> e1 = b - a, e2 = c - a, e3 = d - a;
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
MOPHI_HD inline T hex_volume(const mophi::Real3<T> v[8]) {
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

#endif
