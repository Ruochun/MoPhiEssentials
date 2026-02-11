// DEM device-side helper kernel collection

#ifndef MOPHI_COMPRESSION_CUH
#define MOPHI_COMPRESSION_CUH

#include <common/Defines.hpp>
#include <core/Real3.hpp>

#include <cstdint>
#include <type_traits>
#include <limits>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// A few helper functions
////////////////////////////////////////////////////////////////////////////////

MOPHI_HD inline double clamp01(double t) {
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}
MOPHI_HD inline double _absd(double v) {
    return v < 0 ? -v : v;
}

// ---------- Quantization core (compile-time Bits) ----------

template <unsigned Bits>
MOPHI_HD inline uint64_t quantize_nearest(double v, double L0, double L) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits must be in [1,64]");
    if (L <= 0.0)
        return 0ull;

    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    const double t = clamp01((v - L0) / L);
    const double val = t * static_cast<double>(Nminus1);

    return static_cast<uint64_t>(llround(val));
}

template <unsigned Bits>
MOPHI_HD inline double dequantize_nearest(uint64_t idx, double L0, double L) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits must be in [1,64]");
    if (L <= 0.0)
        return L0;

    constexpr uint64_t mask = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    idx &= mask;
    const double t = (Nminus1 == 0ull) ? 0.0 : static_cast<double>(idx) / static_cast<double>(Nminus1);
    return L0 + t * L;
}

// ---------------- Octahedral (oct) encode/decode for unit direction ----------------
MOPHI_HD inline void oct_encode_unit(double nx, double ny, double nz, double& u01, double& v01) {
    const double invL1 = 1.0 / (_absd(nx) + _absd(ny) + _absd(nz) + 1e-30);
    double x = nx * invL1, y = ny * invL1, z = nz * invL1;
    if (z < 0.0) {
        const double ox = x, oy = y;
        x = (ox >= 0.0) ? (1.0 - _absd(oy)) : (_absd(oy) - 1.0);
        y = (oy >= 0.0) ? (1.0 - _absd(ox)) : (_absd(ox) - 1.0);
    }
    u01 = x * 0.5 + 0.5;
    v01 = y * 0.5 + 0.5;
}

MOPHI_HD inline void oct_decode_to_unit(double u01, double v01, double& nx, double& ny, double& nz) {
    double x = 2.0 * u01 - 1.0;
    double y = 2.0 * v01 - 1.0;
    double z = 1.0 - _absd(x) - _absd(y);
    if (z < 0.0) {
        const double ox = x, oy = y;
        x = (ox >= 0.0) ? (1.0 - _absd(oy)) : (_absd(oy) - 1.0);
        y = (oy >= 0.0) ? (1.0 - _absd(ox)) : (_absd(ox) - 1.0);
    }
    const double len2 = x * x + y * y + z * z;
    const double invL = (len2 > 0.0) ? (1.0 / sqrt(len2)) : 0.0;
    nx = x * invL;
    ny = y * invL;
    nz = z * invL;
}

// ---------------- Bit-quant helpers ----------------
template <unsigned Bits>
MOPHI_HD inline uint64_t quantize01_round(double t01) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits in [1,64]");
    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    const double v = clamp01(t01) * static_cast<double>(Nminus1);
    return static_cast<uint64_t>(llround(v));
}
template <unsigned Bits>
MOPHI_HD inline double dequantize01(uint64_t code) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits in [1,64]");
    constexpr uint64_t mask = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    const uint64_t c = code & mask;
    return (Nminus1 == 0ull) ? 0.0 : static_cast<double>(c) / static_cast<double>(Nminus1);
}

// Log magnitude (by-value params): code 0 = exact zero;
// codes 1..(N-1) map [mag_min, mag_max] logarithmically.
template <unsigned Bits>
MOPHI_HD inline uint64_t quantize_mag_log(double m, double mag_min, double mag_max, double zero_eps) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits in [1,64]");
    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    if (m <= zero_eps)
        return 0ull;
    if (mag_max <= mag_min)
        return 1ull;  // guard
    const double s = clamp01((log(m / mag_min)) / (log(mag_max / mag_min)));
    const double scaled = s * static_cast<double>(Nminus1 - 1ull);
    const uint64_t idx = static_cast<uint64_t>(llround(scaled));
    return 1ull + idx;
}
template <unsigned Bits>
MOPHI_HD inline double dequantize_mag_log(uint64_t code, double mag_min, double mag_max) {
    static_assert(Bits >= 1 && Bits <= 64, "Bits in [1,64]");
    if (code == 0ull)
        return 0.0;
    constexpr uint64_t mask = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    constexpr uint64_t Nminus1 = (Bits == 64) ? UINT64_MAX : ((uint64_t{1} << Bits) - 1ull);
    const uint64_t idx = (code & mask) - 1ull;  // 0..(Nminus1-1)
    if (mag_max <= mag_min)
        return mag_min;
    const double s = (Nminus1 <= 1ull) ? 0.0 : static_cast<double>(idx) / static_cast<double>(Nminus1 - 1ull);
    return mag_min * exp(s * log(mag_max / mag_min));
}

// ---------- Traits for custom compressed types ----------
//
// Specialize this for type T to declare bit-lengths and accessors.

// Linear-scale compression of 3D data
template <typename T>
struct CompressedPointTraits;  // no default; must be specialized

// Helper macro to declare a simple traits specialization when the type
// has fields named XField/YField/ZField (may be bitfields).
#define MOPHI_DECLARE_COMPRESSED_POINT_TRAITS(Type, XField, YField, ZField)                                           \
    template <>                                                                                                       \
    struct CompressedPointTraits<Type> {                                                                              \
        static constexpr unsigned BitsX = Type::kBitsX;                                                               \
        static constexpr unsigned BitsY = Type::kBitsY;                                                               \
        static constexpr unsigned BitsZ = Type::kBitsZ;                                                               \
        static constexpr size_t Align = alignof(Type);                                                                \
        static_assert(BitsX >= 1 && BitsX <= 64 && BitsY >= 1 && BitsY <= 64 && BitsZ >= 1 && BitsZ <= 64,            \
                      "Bit widths must be in [1,64]");                                                                \
        MOPHI_HD static void put_x(Type& dst, uint64_t v) { dst.XField = v; }                                         \
        MOPHI_HD static void put_y(Type& dst, uint64_t v) { dst.YField = v; }                                         \
        MOPHI_HD static void put_z(Type& dst, uint64_t v) { dst.ZField = v; }                                         \
        MOPHI_HD static uint64_t get_x(const Type& src) { return (uint64_t)src.XField; }                              \
        MOPHI_HD static uint64_t get_y(const Type& src) { return (uint64_t)src.YField; }                              \
        MOPHI_HD static uint64_t get_z(const Type& src) { return (uint64_t)src.ZField; }                              \
        MOPHI_HD static uint64_t quant_x(double x, double L0, double L) { return quantize_nearest<BitsX>(x, L0, L); } \
        MOPHI_HD static uint64_t quant_y(double y, double L0, double L) { return quantize_nearest<BitsY>(y, L0, L); } \
        MOPHI_HD static uint64_t quant_z(double z, double L0, double L) { return quantize_nearest<BitsZ>(z, L0, L); } \
        MOPHI_HD static double dequant_x(uint64_t ix, double L0, double L) {                                          \
            return dequantize_nearest<BitsX>(ix, L0, L);                                                              \
        }                                                                                                             \
        MOPHI_HD static double dequant_y(uint64_t iy, double L0, double L) {                                          \
            return dequantize_nearest<BitsY>(iy, L0, L);                                                              \
        }                                                                                                             \
        MOPHI_HD static double dequant_z(uint64_t iz, double L0, double L) {                                          \
            return dequantize_nearest<BitsZ>(iz, L0, L);                                                              \
        }                                                                                                             \
    };

// Then for log-scale compression of 3D data...
template <typename T>
struct CompressedLogscaleTraits;

// Macro: Type provides kBitsMag/kBitsU/kBitsV; we bind field names + snake_case accessors.
#define MOPHI_DECLARE_LOGSCALE_TRAITS(Type, MagField, UField, VField)                                          \
    template <>                                                                                                \
    struct CompressedLogscaleTraits<Type> {                                                                    \
        static constexpr unsigned BitsMag = Type::kBitsMag;                                                    \
        static constexpr unsigned BitsU = Type::kBitsU;                                                        \
        static constexpr unsigned BitsV = Type::kBitsV;                                                        \
        static constexpr size_t Align = alignof(Type);                                                         \
        static_assert(BitsMag >= 1 && BitsMag <= 64 && BitsU >= 1 && BitsU <= 64 && BitsV >= 1 && BitsV <= 64, \
                      "Bit widths must be in [1,64]");                                                         \
        MOPHI_HD static void put_mag(Type& d, uint64_t v) { d.MagField = v; }                                  \
        MOPHI_HD static void put_u(Type& d, uint64_t v) { d.UField = v; }                                      \
        MOPHI_HD static void put_v(Type& d, uint64_t v) { d.VField = v; }                                      \
        MOPHI_HD static uint64_t get_mag(const Type& s) { return (uint64_t)s.MagField; }                       \
        MOPHI_HD static uint64_t get_u(const Type& s) { return (uint64_t)s.UField; }                           \
        MOPHI_HD static uint64_t get_v(const Type& s) { return (uint64_t)s.VField; }                           \
    }

// Traits of the compressed types; the underlying structs are defined in common/Defines.hpp
MOPHI_DECLARE_COMPRESSED_POINT_TRAITS(mophi::CompLinear3D_128Bit, x, y, z);
MOPHI_DECLARE_LOGSCALE_TRAITS(mophi::CompLog3D_64Bit, m, u, v);

// ---------- Point-wise compress/decompress using traits ----------

// Compress one point: (p) + (LBF,size) pointers -> out (compressed)
template <typename CompT, typename TFP = double>
MOPHI_HD inline void CompressPoint_T(const mophi::Real3<TFP>* p,
                                     const mophi::Real3d& LBF,
                                     const mophi::Real3d& size,
                                     CompT* out) {
    using Tr = CompressedPointTraits<CompT>;
    const uint64_t ix = Tr::quant_x(static_cast<double>(p->x()), LBF.x(), size.x());
    const uint64_t iy = Tr::quant_y(static_cast<double>(p->y()), LBF.y(), size.y());
    const uint64_t iz = Tr::quant_z(static_cast<double>(p->z()), LBF.z(), size.z());
    Tr::put_x(*out, ix);
    Tr::put_y(*out, iy);
    Tr::put_z(*out, iz);
}

// Decompress one point: (in) + (LBF,size) pointers -> out Real3<TFP>
template <typename CompT, typename TFP = double>
MOPHI_HD inline void DecompressPoint_T(const CompT* in,
                                       const mophi::Real3d& LBF,
                                       const mophi::Real3d& size,
                                       mophi::Real3<TFP>* out) {
    using Tr = CompressedPointTraits<CompT>;
    const double x = Tr::dequant_x(Tr::get_x(*in), LBF.x(), size.x());
    const double y = Tr::dequant_y(Tr::get_y(*in), LBF.y(), size.y());
    const double z = Tr::dequant_z(Tr::get_z(*in), LBF.z(), size.z());
    *out = mophi::Real3<TFP>(static_cast<TFP>(x), static_cast<TFP>(y), static_cast<TFP>(z));
}

template <typename CompT, typename TFP = double>
MOPHI_HD inline void CompressLogscale_T(const mophi::Real3<TFP>* v,
                                        double mag_min,
                                        double mag_max,
                                        double zero_eps,
                                        CompT* out) {
    using Tr = CompressedLogscaleTraits<CompT>;
    const double vx = v->x(), vy = v->y(), vz = v->z();
    const double m = sqrt(vx * vx + vy * vy + vz * vz);
    const uint64_t mcode = quantize_mag_log<Tr::BitsMag>(m, mag_min, mag_max, zero_eps);

    uint64_t ucode = 0, vcode = 0;
    if (mcode != 0ull) {
        const double invm = 1.0 / m;
        double u01, v01;
        oct_encode_unit(vx * invm, vy * invm, vz * invm, u01, v01);
        ucode = quantize01_round<Tr::BitsU>(u01);
        vcode = quantize01_round<Tr::BitsV>(v01);
    }
    Tr::put_mag(*out, mcode);
    Tr::put_u(*out, ucode);
    Tr::put_v(*out, vcode);
}

template <typename CompT, typename TFP = double>
MOPHI_HD inline void DecompressLogscale_T(const CompT* in,
                                          double mag_min,
                                          double mag_max, /* zero_eps not needed here */
                                          mophi::Real3<TFP>* v_out) {
    using Tr = CompressedLogscaleTraits<CompT>;
    const uint64_t mcode = Tr::get_mag(*in);
    if (mcode == 0ull) {
        *v_out = mophi::Real3<TFP>(TFP(0), TFP(0), TFP(0));
        return;
    }
    const double m = dequantize_mag_log<Tr::BitsMag>(mcode, mag_min, mag_max);
    const double u01 = dequantize01<Tr::BitsU>(Tr::get_u(*in));
    const double v01 = dequantize01<Tr::BitsV>(Tr::get_v(*in));
    double nx, ny, nz;
    oct_decode_to_unit(u01, v01, nx, ny, nz);
    *v_out = mophi::Real3<TFP>(static_cast<TFP>(m * nx), static_cast<TFP>(m * ny), static_cast<TFP>(m * nz));
}

#endif
