//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_MESH_HPP
#define MOPHI_MESH_HPP

#include "../utils/robin_hood.h"
#include "VariableTypes.hpp"
#include "SharedStructs.hpp"
#include "../core/DataClasses.hpp"
#include "../core/Real3.hpp"
#include "../core/Quaternion.hpp"
#include "../kernels/Compression.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace mophi {

//////////////////////////////////////////////////////
// Mesh data structures for MoPhi
// The meshes themselves are cached in the main solver
//////////////////////////////////////////////////////

enum class CellType : uint8_t { Tet, Hex };

struct PartConnectivity {
    // Owned cells of this part (no ghosts here)
    std::vector<std::array<nodeID_t, 4>> tets;
    std::vector<std::array<nodeID_t, 10>> tet10s;
    std::vector<std::array<nodeID_t, 8>> hexes;
    // Per-element region/BC tags
    std::vector<meshTag_t> tetTags, tet10Tags, hexTags;
};

struct PartGeometry {
    // Localized node coords: OWNED nodes come first, then HALO nodes (compact!)
    std::vector<Real3d> nodes;
    uNodeID_t nOwnedNodes = 0;  // nodes[0 .. nOwnedNodes) are owned; rest are halo
};

// Halo bookkeeping for intra-node (multi-GPU) exchange now; MPI later if needed.
struct HaloDesc {
    // Node/global ↔ local maps
    std::vector<nodeIDGlobal_t> localToGlobalNode;
    robin_hood::unordered_flat_map<nodeIDGlobal_t, nodeID_t> globalToLocalNode;

    // Halo neighbors (GPU IDs) and index lists (compact, for pack/unpack)
    struct Neighbor {
        meshPart_t partID;                // 0..N-1 (single host, multiple GPUs)
        std::vector<nodeID_t> sendNodes;  // local (owned) nodes to send to this neighbor
        std::vector<nodeID_t> recvNodes;  // local (halo) nodes to receive from this neighbor
    };
    std::vector<Neighbor> neighbors;
};

struct Mesh {
    meshPart_t partID = -1;

    // Localized, compact, GPU-friendly topology+geom for this part
    PartConnectivity topo;
    PartGeometry geom;

    // Optional: global IDs for cells if you keep them
    std::vector<nodeIDGlobal_t> localToGlobalCell;

    // Halo description
    HaloDesc halo;

    // The order the mesh is loaded
    unsigned loadOrder = 0;

    // Quick counts
    // Owned means strictly I manage
    uNodeID_t NumOwnedNodes() const { return geom.nOwnedNodes; }
    // Local means all that I store, including halo
    uNodeID_t NumLocalNodes() const { return (uNodeID_t)geom.nodes.size(); }
    // Number of geos
    uNodeID_t NumOwnedCells() const { return (uNodeID_t)(topo.tets.size() + topo.tet10s.size() + topo.hexes.size()); }
    uNodeID_t NumOwnedTets() const { return topo.tets.size(); }
    uNodeID_t NumOwnedTet10s() const { return topo.tet10s.size(); }
    uNodeID_t NumOwnedHexes() const { return topo.hexes.size(); }

    // =========================================================================
    // Geometric transformation utilities
    // =========================================================================

    /// Translate all nodes by the given offset vector.
    void Translate(const Real3d& offset) {
        for (auto& n : geom.nodes)
            n = n + offset;
    }

    /// Rotate all nodes about the origin using the given quaternion.
    /// @param q Unit quaternion representing the desired rotation.  Passing a
    ///          non-unit quaternion will scale the result and produce incorrect
    ///          geometry; normalize @p q before calling if necessary.
    void Rotate(const Quatd& q) {
        for (auto& n : geom.nodes)
            n = q.Rotate(n);
    }

    /// Rotate all nodes about the origin and then translate by offset.
    /// @param q Unit quaternion representing the desired rotation.  Passing a
    ///          non-unit quaternion will produce incorrect geometry; normalize
    ///          @p q before calling if necessary.
    void RotateAndTranslate(const Quatd& q, const Real3d& offset) {
        for (auto& n : geom.nodes)
            n = q.Rotate(n) + offset;
    }

    /// Scale all nodes uniformly by a scalar factor.
    void Scale(double factor) {
        for (auto& n : geom.nodes)
            n = n * factor;
    }

    /// Scale all nodes independently along each axis.
    void Scale(double sx, double sy, double sz) {
        for (auto& n : geom.nodes) {
            n.x() *= sx;
            n.y() *= sy;
            n.z() *= sz;
        }
    }

    /// Mirror all nodes across the plane defined by a normal direction and a
    /// point on the plane.  The normal is normalized internally so any
    /// non-zero direction vector may be supplied.
    /// Reflection formula: x' = x - 2*((x-p)·n̂)*n̂
    void Mirror(const Real3d& normal, const Real3d& point = Real3d(0, 0, 0)) {
        Real3d n = normal.GetNormalized();
        for (auto& nd : geom.nodes) {
            Real3d d = nd - point;
            double proj = d ^ n;  // dot product
            nd = nd - n * (2.0 * proj);
        }
    }
};

//////////////////////////////////////////////////////////////
// Mesh soup for decomposed mesh, easier to use on GPU
// Workers generate and use the mesh soup, and it's
// a TEMPORARY sttucture in our solver
//////////////////////////////////////////////////////////////

// ---- Cells (Tet / Hex) ----

//// TODO: Questionable if we should precompute these
using Real = double;
struct TetGeom {
    Real volume;
    Real3d centroid;
    // For P1 FEM it’s handy to precompute physical ∇N_i (constant in tet)
    Real3d gradN[4];  // ∑ gradN[i] == 0
};

//// TODO: Questionable if we should precompute these
struct HexGeom {
    Real volume;      // via stable 5-tet fan
    Real3d centroid;  // mean of 8 vertices
    // (Gradients depend on mapping; skip for now)
};

// ---- Faces (triangulated) ----
struct FaceTopo {
    nodeID_t a, b, c;      // local node ids (tri)
    nodeID_t cellOwned;    // unified local cell id (see mapping below)
    nodeID_t cellOther;    // neighbor cell id on this part, or (nodeID_t)-1 if none
    int16_t neighborPart;  // >=0 interface, -1 not interface
    int16_t bcTag;         // >0 boundary code, 0 otherwise
};

//// TODO: Questionable if we should precompute these
struct FaceGeom {
    Real area;
    Real3d normal;  // unit, outward from cellOwned
    Real3d centroid;
};

// ---- Unified cell indexing (for CSR and face adjacency) ----
// localCellId in [0 .. nTets+nHexes):
//   if id < nTets -> tet index = id
//   else           -> hex index = id - nTets

struct MeshSoup {
    // Cells
    std::vector<TetTopo> tetsTopo;
    std::vector<TetGeom> tetsGeom;

    std::vector<HexTopo> hexesTopo;
    std::vector<HexGeom> hexesGeom;

    // Faces
    std::vector<FaceTopo> facesTopo;
    std::vector<FaceGeom> facesGeom;

    // Optional: cell -> faces CSR over unified local cell ids
    std::vector<uNodeID_t> cellFaceOffsets;  // size = nCells+1
    std::vector<uNodeID_t> cellFaceIndices;  // concatenated face indices

    // Counts
    uNodeID_t NumTets() const { return (uNodeID_t)tetsTopo.size(); }
    uNodeID_t NumHexes() const { return (uNodeID_t)hexesTopo.size(); }
    uNodeID_t NumCells() const { return NumTets() + NumHexes(); }
    uNodeID_t NumFaces() const { return (uNodeID_t)facesTopo.size(); }
};

struct FaceInfo {
    int neighborPart;
    meshTag_t bcTag;
};
using FaceClassifier = std::function<std::optional<FaceInfo>(std::array<nodeIDGlobal_t, 3> sortedGlobalTri)>;

struct SoupOptions {
    bool computeMetrics = true;
    bool buildCellFaceCSR = true;  // fill CSR
    FaceClassifier classifier;     // optional override
};

struct LocalTriKey {
    nodeID_t i, j, k;
    bool operator==(const LocalTriKey& o) const noexcept { return i == o.i && j == o.j && k == o.k; }
};
struct LocalTriKeyHash {
    size_t operator()(const LocalTriKey& t) const noexcept {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        mix((uint64_t)t.i);
        mix((uint64_t)t.j);
        mix((uint64_t)t.k);
        return h;
    }
};
inline LocalTriKey key_sorted(nodeID_t a, nodeID_t b, nodeID_t c) {
    if (a > b)
        std::swap(a, b);
    if (b > c)
        std::swap(b, c);
    if (a > b)
        std::swap(a, b);
    return {a, b, c};
}
inline std::array<nodeIDGlobal_t, 3> sorted_global_tri(const Mesh& m, nodeID_t a, nodeID_t b, nodeID_t c) {
    nodeIDGlobal_t ga = m.halo.localToGlobalNode[(size_t)a];
    nodeIDGlobal_t gb = m.halo.localToGlobalNode[(size_t)b];
    nodeIDGlobal_t gc = m.halo.localToGlobalNode[(size_t)c];
    if (ga > gb)
        std::swap(ga, gb);
    if (gb > gc)
        std::swap(gb, gc);
    if (ga > gb)
        std::swap(ga, gb);
    return {ga, gb, gc};
}

// hex face triangulation (consistent)
inline void hex_face_tris(const mophi::nodeID_t* h, int faceIdx, std::array<std::array<mophi::nodeID_t, 3>, 2>& out) {
    // faces: (0)0123 (1)0145 (2)0264 (3)1375 (4)2367 (5)4567
    switch (faceIdx) {
        case 0:
            out = {{{h[0], h[1], h[3]}, {h[0], h[3], h[2]}}};
            break;
        case 1:
            out = {{{h[0], h[4], h[5]}, {h[0], h[5], h[1]}}};
            break;
        case 2:
            out = {{{h[0], h[2], h[6]}, {h[0], h[6], h[4]}}};
            break;
        case 3:
            out = {{{h[1], h[5], h[7]}, {h[1], h[7], h[3]}}};
            break;
        case 4:
            out = {{{h[2], h[3], h[7]}, {h[2], h[7], h[6]}}};
            break;
        case 5:
            out = {{{h[4], h[6], h[7]}, {h[4], h[7], h[5]}}};
            break;
        default:
            out = {{{0, 0, 0}, {0, 0, 0}}};
            break;
    }
}

// ---- main MeshSoup builder ----
inline MeshSoup BuildMeshSoup(const Mesh& mesh, const SoupOptions& opt) {
    MeshSoup S;

    // 0) Copy cell topology into flat vectors and compute geometry
    S.tetsTopo.reserve(mesh.topo.tets.size());
    S.tetsGeom.resize(mesh.topo.tets.size());
    for (size_t c = 0; c < mesh.topo.tets.size(); ++c) {
        const auto& t = mesh.topo.tets[c];
        S.tetsTopo.push_back(TetTopo{{t[0], t[1], t[2], t[3]}});
        if (opt.computeMetrics) {
            const Real3d a = mesh.geom.nodes[t[0]];
            const Real3d b = mesh.geom.nodes[t[1]];
            const Real3d c3 = mesh.geom.nodes[t[2]];
            const Real3d d = mesh.geom.nodes[t[3]];
            S.tetsGeom[c].centroid = tet_centroid(a, b, c3, d);
            S.tetsGeom[c].volume = tet_volume(a, b, c3, d);
            tet_gradN_phys(a, b, c3, d, S.tetsGeom[c].gradN);
        }
    }

    S.hexesTopo.reserve(mesh.topo.hexes.size());
    S.hexesGeom.resize(mesh.topo.hexes.size());
    for (size_t c = 0; c < mesh.topo.hexes.size(); ++c) {
        const auto& h = mesh.topo.hexes[c];
        S.hexesTopo.push_back(HexTopo{{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]}});
        if (opt.computeMetrics) {
            Real3d v[8];
            for (int i = 0; i < 8; ++i)
                v[i] = mesh.geom.nodes[h[i]];
            S.hexesGeom[c].centroid = hex_centroid(v);
            S.hexesGeom[c].volume = hex_volume(v);
        }
    }

    const uNodeID_t nT = S.NumTets();
    // helper to map (unified) local cell id
    auto tetIdToCell = [&](uNodeID_t it) -> nodeID_t { return (nodeID_t)it; };
    auto hexIdToCell = [&](uNodeID_t ih) -> nodeID_t { return (nodeID_t)(nT + ih); };

    // 1) Deduplicate faces across all owned cells and classify
    std::unordered_map<LocalTriKey, uNodeID_t, LocalTriKeyHash> fmap;
    fmap.reserve((S.NumCells()) * 2);

    auto add_face = [&](nodeID_t a, nodeID_t b, nodeID_t c, nodeID_t ownerCell) {
        const auto k = key_sorted(a, b, c);
        auto it = fmap.find(k);
        if (it == fmap.end()) {
            FaceTopo ft{};
            ft.a = a;
            ft.b = b;
            ft.c = c;
            ft.cellOwned = ownerCell;
            ft.cellOther = (nodeID_t)(-1);
            ft.neighborPart = -1;
            ft.bcTag = 0;
            FaceGeom fg{};  // fill later
            uNodeID_t idx = (uNodeID_t)S.facesTopo.size();
            S.facesTopo.push_back(ft);
            S.facesGeom.push_back(fg);
            fmap.emplace(k, idx);
        } else {
            // second time -> interior
            S.facesTopo[it->second].cellOther = ownerCell;
        }
    };

    // tets: 4 faces each
    for (uNodeID_t it = 0; it < nT; ++it) {
        const auto& t = S.tetsTopo[it].v;
        const nodeID_t cell = tetIdToCell(it);
        add_face(t[0], t[1], t[2], cell);
        add_face(t[0], t[1], t[3], cell);
        add_face(t[0], t[2], t[3], cell);
        add_face(t[1], t[2], t[3], cell);
    }
    // hexes: 6 faces, each triangulated into 2 tris
    for (uNodeID_t ih = 0; ih < S.NumHexes(); ++ih) {
        const auto& h = S.hexesTopo[ih].v;
        std::array<std::array<nodeID_t, 3>, 2> tris;
        const nodeID_t cell = hexIdToCell(ih);
        for (int f = 0; f < 6; ++f) {
            hex_face_tris(S.hexesTopo[ih].v, f, tris);
            add_face(tris[0][0], tris[0][1], tris[0][2], cell);
            add_face(tris[1][0], tris[1][1], tris[1][2], cell);
        }
    }

    // 2) Classify single-sided faces (interface/boundary) and compute face geom
    auto is_halo = [&](nodeID_t l) { return l >= mesh.geom.nOwnedNodes; };

    for (size_t fi = 0; fi < S.facesTopo.size(); ++fi) {
        auto& FT = S.facesTopo[fi];
        auto& FG = S.facesGeom[fi];

        // classification (prefer classifier if provided)
        FaceInfo info{-1, 0};
        bool have = false;
        if (opt.classifier) {
            auto keyG = sorted_global_tri(mesh, FT.a, FT.b, FT.c);
            if (auto r = opt.classifier(keyG)) {
                info = *r;
                have = true;
            }
        }
        if (have) {
            FT.neighborPart = (int16_t)info.neighborPart;
            FT.bcTag = (int16_t)info.bcTag;
        } else {
            if (FT.cellOther == (nodeID_t)(-1)) {
                if (is_halo(FT.a) || is_halo(FT.b) || is_halo(FT.c))
                    FT.neighborPart = 0;  // interface
                else {
                    FT.neighborPart = -1;
                    FT.bcTag = 0;
                }  // physical boundary (untagged)
            }
        }

        if (opt.computeMetrics) {
            const Real3d A = mesh.geom.nodes[FT.a];
            const Real3d B = mesh.geom.nodes[FT.b];
            const Real3d C = mesh.geom.nodes[FT.c];
            FG.centroid = tri_centroid(A, B, C);
            Real3d n = (B - A) % (C - A);
            FG.area = (Real)0.5 * n.Length();
            if (FG.area > (Real)0)
                n = n * ((Real)1.0 / ((Real)2.0 * FG.area));  // unit

            // outward from owner cell centroid
            Real3d cOwner;
            if (FT.cellOwned < nT)
                cOwner = S.tetsGeom[FT.cellOwned].centroid;
            else
                cOwner = S.hexesGeom[FT.cellOwned - nT].centroid;
            if ((n ^ (cOwner - FG.centroid)) > (Real)0)
                n = n * (Real)(-1.0);
            FG.normal = n;
        }
    }

    // 3) Optional CSR: cell -> faces over unified ids
    if (opt.buildCellFaceCSR) {
        const uNodeID_t nC = S.NumCells();
        std::vector<uNodeID_t> counts(nC, 0);
        for (uNodeID_t fi = 0; fi < S.NumFaces(); ++fi) {
            const auto& FT = S.facesTopo[fi];
            counts[FT.cellOwned]++;
            if (FT.cellOther != (nodeID_t)(-1))
                counts[FT.cellOther]++;
        }
        S.cellFaceOffsets.resize(nC + 1);
        uNodeID_t off = 0;
        for (uNodeID_t c = 0; c < nC; ++c) {
            S.cellFaceOffsets[c] = off;
            off += counts[c];
        }
        S.cellFaceOffsets[nC] = off;
        S.cellFaceIndices.resize(off);
        std::fill(counts.begin(), counts.end(), 0);

        for (uNodeID_t fi = 0; fi < S.NumFaces(); ++fi) {
            const auto& FT = S.facesTopo[fi];
            auto place = [&](nodeID_t c) {
                uNodeID_t pos = S.cellFaceOffsets[(size_t)c] + counts[(size_t)c]++;
                S.cellFaceIndices[(size_t)pos] = fi;
            };
            place(FT.cellOwned);
            if (FT.cellOther != (nodeID_t)(-1))
                place(FT.cellOther);
        }
    }

    return S;
}


// =============================================================================
// SurfaceMesh: triangle surface mesh data structure
// =============================================================================

/// Minimum cross-product length below which a computed face normal is left as
/// a zero vector (degenerate-triangle guard).
constexpr double kMeshNormalLengthEps = 1e-15;

/// Fraction of the bounding-box diagonal used as the vertex-welding tolerance
/// in SurfaceMesh::IsWatertight().  Vertices within this fraction of the
/// diagonal are treated as geometrically identical regardless of their raw
/// integer indices — this handles the duplicate vertices produced by STL file
/// loading.
constexpr double kMeshVertexWeldFactor = 1e-6;

/// Describes one adjacency entry for a triangle: its neighbor and the shared
/// oriented-edge info.  Built by SurfaceMesh::BuildAdjacencyWithEdgeInfo().
struct EdgeAdjInfo {
    size_t nbr;       ///< Index of the neighboring triangle
    int va;           ///< Directed edge start vertex (as in the current triangle)
    int vb;           ///< Directed edge end vertex (as in the current triangle)
    bool oriented_ok; ///< true if the neighbor winding is consistent (edge reversed)
};

/// A surface triangle mesh: vertices, triangular faces, optional per-face/-vertex
/// normals, and optional UV texture coordinates.
///
/// The data layout mirrors standard OBJ/PLY/STL conventions:
///   - vertices          – vertex positions (shared pool)
///   - faces             – triangle vertex indices (3 per face, 0-based into vertices)
///   - normals           – shared normal pool (per-vertex OR per-face depending on source)
///   - faceNormalIndices – one triple per face indexing into normals (same length as faces)
///   - uvs               – shared UV pool (z component is 0 for proper 2-D UV)
///   - faceUVIndices     – one triple per face indexing into uvs
struct SurfaceMesh {
    std::vector<Real3d> vertices;
    std::vector<std::array<int, 3>> faces;
    std::vector<Real3d> normals;
    std::vector<std::array<int, 3>> faceNormalIndices;
    std::vector<Real3d> uvs;
    std::vector<std::array<int, 3>> faceUVIndices;

    // -------------------------------------------------------------------------
    // Counts and queries
    // -------------------------------------------------------------------------
    size_t NumFaces() const { return faces.size(); }
    size_t NumVertices() const { return vertices.size(); }
    bool HasNormals() const { return !normals.empty() && !faceNormalIndices.empty(); }
    bool HasUVs() const { return !uvs.empty() && !faceUVIndices.empty(); }

    void Clear() {
        vertices.clear();
        faces.clear();
        normals.clear();
        faceNormalIndices.clear();
        uvs.clear();
        faceUVIndices.clear();
    }

    // -------------------------------------------------------------------------
    // Geometric transformation utilities (same API as Mesh above)
    // -------------------------------------------------------------------------

    /// Translate all vertices by the given offset.
    void Translate(const Real3d& offset) {
        for (auto& v : vertices)
            v = v + offset;
    }

    /// Scale all vertices uniformly.
    void Scale(double factor) {
        for (auto& v : vertices)
            v = v * factor;
    }

    /// Scale all vertices independently per axis.
    void Scale(double sx, double sy, double sz) {
        for (auto& v : vertices) {
            v.x() *= sx;
            v.y() *= sy;
            v.z() *= sz;
        }
    }

    /// Rotate all vertices (and normals) about the origin using a unit quaternion.
    void Rotate(const Quatd& q) {
        for (auto& v : vertices)
            v = q.Rotate(v);
        for (auto& n : normals)
            n = q.Rotate(n);
    }

    /// Rotate all vertices (and normals) then translate.
    void RotateAndTranslate(const Quatd& q, const Real3d& offset) {
        for (auto& v : vertices)
            v = q.Rotate(v) + offset;
        for (auto& n : normals)
            n = q.Rotate(n);
    }

    /// Mirror all vertices across the plane defined by a normal direction and a
    /// reference point.  Normals are reflected too (mirroring flips orientation).
    void Mirror(const Real3d& normal, const Real3d& point = Real3d(0, 0, 0)) {
        Real3d n = normal.GetNormalized();
        for (auto& nd : vertices) {
            Real3d d = nd - point;
            double proj = d ^ n;
            nd = nd - n * (2.0 * proj);
        }
        for (auto& nrm : normals) {
            double proj = nrm ^ n;
            nrm = nrm - n * (2.0 * proj);
        }
    }

    // -------------------------------------------------------------------------
    // Mesh analysis utilities
    // -------------------------------------------------------------------------

    /// Compute one geometric normal per face from vertex winding and store them
    /// in normals/faceNormalIndices (overwrites any previous normals).
    void ComputeFaceNormals() {
        normals.clear();
        faceNormalIndices.clear();
        normals.reserve(faces.size());
        faceNormalIndices.reserve(faces.size());
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            Real3d e1 = vertices[f[1]] - vertices[f[0]];
            Real3d e2 = vertices[f[2]] - vertices[f[0]];
            Real3d nm = e1 % e2;
            double len = nm.Length();
            if (len > kMeshNormalLengthEps)
                nm = nm * (1.0 / len);
            normals.push_back(nm);
            faceNormalIndices.push_back({(int)i, (int)i, (int)i});
        }
    }

    /// Test whether the mesh is watertight (no boundary or non-manifold edges).
    ///
    /// First checks using raw vertex indices; if issues are found, falls back to
    /// tolerance-based vertex welding (quantisation at 1e-6 of bounding-box
    /// diagonal) to handle duplicate vertices produced by STL file loading.
    ///
    /// @param boundary_edges_out     Receives the number of boundary edges found.
    /// @param nonmanifold_edges_out  Receives the number of non-manifold edges found.
    /// @returns true if the mesh is watertight.
    bool IsWatertight(size_t* boundary_edges_out = nullptr,
                      size_t* nonmanifold_edges_out = nullptr) const {
        if (faces.empty()) {
            if (boundary_edges_out)
                *boundary_edges_out = 0;
            if (nonmanifold_edges_out)
                *nonmanifold_edges_out = 0;
            return true;
        }

        // Helper: count boundary/non-manifold from a canonical index array.
        auto count_edges = [&](const std::vector<size_t>& canon,
                               size_t& boundary, size_t& nonmanifold) {
            std::map<std::pair<size_t, size_t>, size_t> edge_counts;
            for (const auto& f : faces) {
                int a0 = f[0], b0 = f[1], c0 = f[2];
                if (a0 < 0 || b0 < 0 || c0 < 0)
                    continue;
                size_t a = (!canon.empty()) ? canon[(size_t)a0] : (size_t)a0;
                size_t b = (!canon.empty()) ? canon[(size_t)b0] : (size_t)b0;
                size_t c = (!canon.empty()) ? canon[(size_t)c0] : (size_t)c0;
                if (a == b || b == c || c == a)
                    continue;
                edge_counts[{std::min(a, b), std::max(a, b)}]++;
                edge_counts[{std::min(b, c), std::max(b, c)}]++;
                edge_counts[{std::min(c, a), std::max(c, a)}]++;
            }
            boundary = 0;
            nonmanifold = 0;
            for (const auto& kv : edge_counts) {
                if (kv.second == 1)
                    ++boundary;
                else if (kv.second > 2)
                    ++nonmanifold;
            }
        };

        // Pass 1: raw indices.
        size_t boundary1 = 0, nonmanifold1 = 0;
        count_edges({}, boundary1, nonmanifold1);

        if (boundary1 == 0 && nonmanifold1 == 0) {
            if (boundary_edges_out)
                *boundary_edges_out = 0;
            if (nonmanifold_edges_out)
                *nonmanifold_edges_out = 0;
            return true;
        }

        // Pass 2: quantisation-based vertex welding (handles duplicate verts from STL).
        if (!vertices.empty()) {
            double bb_min[3] = {vertices[0].x(), vertices[0].y(), vertices[0].z()};
            double bb_max[3] = {vertices[0].x(), vertices[0].y(), vertices[0].z()};
            for (const auto& v : vertices) {
                if (v.x() < bb_min[0]) bb_min[0] = v.x();
                if (v.y() < bb_min[1]) bb_min[1] = v.y();
                if (v.z() < bb_min[2]) bb_min[2] = v.z();
                if (v.x() > bb_max[0]) bb_max[0] = v.x();
                if (v.y() > bb_max[1]) bb_max[1] = v.y();
                if (v.z() > bb_max[2]) bb_max[2] = v.z();
            }
            double diag2 = 0;
            for (int k = 0; k < 3; ++k)
                diag2 += (bb_max[k] - bb_min[k]) * (bb_max[k] - bb_min[k]);
            double eps = std::sqrt(diag2) * kMeshVertexWeldFactor;
            if (eps <= 0.0)
                eps = 1e-10;

            auto qc = [&](double v, int axis) -> int64_t {
                return static_cast<int64_t>(std::floor((v - bb_min[axis]) / eps));
            };

            using Key3 = std::array<int64_t, 3>;
            std::map<Key3, size_t> grid;
            std::vector<size_t> canon(vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                Key3 k = {qc(vertices[i].x(), 0), qc(vertices[i].y(), 1), qc(vertices[i].z(), 2)};
                auto it = grid.find(k);
                if (it == grid.end()) {
                    grid[k] = i;
                    canon[i] = i;
                } else {
                    canon[i] = it->second;
                }
            }

            size_t boundary2 = 0, nonmanifold2 = 0;
            count_edges(canon, boundary2, nonmanifold2);

            if (boundary_edges_out)
                *boundary_edges_out = boundary2;
            if (nonmanifold_edges_out)
                *nonmanifold_edges_out = nonmanifold2;
            return boundary2 == 0 && nonmanifold2 == 0;
        }

        if (boundary_edges_out)
            *boundary_edges_out = boundary1;
        if (nonmanifold_edges_out)
            *nonmanifold_edges_out = nonmanifold1;
        return false;
    }

    /// Compute volume, center of mass and diagonal inertia components (Ixx, Iyy, Izz)
    /// in the center-of-mass frame assuming unit density and a solid (watertight) mesh.
    ///
    /// Uses the divergence-theorem formulation; sign correction is applied if
    /// triangle winding produces a negative signed volume.
    void ComputeMassProperties(double& volume, Real3d& center, Real3d& inertia) const {
        Real3d dummy(0, 0, 0);
        ComputeMassProperties(volume, center, inertia, dummy);
    }

    /// Compute volume, center of mass, diagonal inertia (Ixx, Iyy, Izz), and
    /// off-diagonal inertia products (Ixy, Iyz, Izx) in the center-of-mass frame
    /// assuming unit density and a solid (watertight) mesh.
    void ComputeMassProperties(double& volume, Real3d& center, Real3d& inertia,
                               Real3d& inertia_products) const {
        double vol = 0.0;
        double mx = 0.0, my = 0.0, mz = 0.0;
        double ix2 = 0.0, iy2 = 0.0, iz2 = 0.0;
        double ixy = 0.0, iyz = 0.0, izx = 0.0;

        for (const auto& f : faces) {
            const Real3d& a = vertices[(size_t)f[0]];
            const Real3d& b = vertices[(size_t)f[1]];
            const Real3d& c = vertices[(size_t)f[2]];

            // Signed volume contribution: dot(a, cross(b, c)) / 6
            Real3d bc = b % c;
            double v = (a ^ bc) / 6.0;

            vol += v;
            mx += v * (a.x() + b.x() + c.x()) / 4.0;
            my += v * (a.y() + b.y() + c.y()) / 4.0;
            mz += v * (a.z() + b.z() + c.z()) / 4.0;

            const double ax = a.x(), ay = a.y(), az = a.z();
            const double bx = b.x(), by = b.y(), bz = b.z();
            const double cx = c.x(), cy = c.y(), cz = c.z();

            const double f1x = ax * ax + bx * bx + cx * cx + ax * bx + bx * cx + cx * ax;
            const double f1y = ay * ay + by * by + cy * cy + ay * by + by * cy + cy * ay;
            const double f1z = az * az + bz * bz + cz * cz + az * bz + bz * cz + cz * az;

            ix2 += v * f1x / 10.0;
            iy2 += v * f1y / 10.0;
            iz2 += v * f1z / 10.0;

            const double fxy = 2.0 * (ax * ay + bx * by + cx * cy) +
                               (ax * by + ay * bx + bx * cy + by * cx + cx * ay + cy * ax);
            const double fyz = 2.0 * (ay * az + by * bz + cy * cz) +
                               (ay * bz + az * by + by * cz + bz * cy + cy * az + cz * ay);
            const double fzx = 2.0 * (az * ax + bz * bx + cz * cx) +
                               (az * bx + ax * bz + bz * cx + bx * cz + cz * ax + cx * az);

            ixy += v * fxy / 20.0;
            iyz += v * fyz / 20.0;
            izx += v * fzx / 20.0;
        }

        if (vol == 0.0) {
            volume = 0.0;
            center = Real3d(0, 0, 0);
            inertia = Real3d(0, 0, 0);
            inertia_products = Real3d(0, 0, 0);
            return;
        }

        // Correct sign if winding produces negative volume.
        if (vol < 0.0) {
            vol = -vol;
            mx = -mx;
            my = -my;
            mz = -mz;
            ix2 = -ix2;
            iy2 = -iy2;
            iz2 = -iz2;
            ixy = -ixy;
            iyz = -iyz;
            izx = -izx;
        }

        const double cx_ = mx / vol;
        const double cy_ = my / vol;
        const double cz_ = mz / vol;

        // Inertia about origin, then shift to CoM via parallel-axis theorem.
        double Ixx = iy2 + iz2;
        double Iyy = ix2 + iz2;
        double Izz = ix2 + iy2;
        double Ixy = -ixy;
        double Iyz = -iyz;
        double Izx = -izx;

        Ixx -= vol * (cy_ * cy_ + cz_ * cz_);
        Iyy -= vol * (cx_ * cx_ + cz_ * cz_);
        Izz -= vol * (cx_ * cx_ + cy_ * cy_);
        Ixy += vol * cx_ * cy_;
        Iyz += vol * cy_ * cz_;
        Izx += vol * cz_ * cx_;

        volume = vol;
        center = Real3d(cx_, cy_, cz_);
        inertia = Real3d(Ixx, Iyy, Izz);
        inertia_products = Real3d(Ixy, Iyz, Izx);
    }

    /// Build per-triangle adjacency with oriented shared-edge information.
    ///
    /// Only manifold edges (shared by exactly 2 triangles) produce adjacency
    /// entries; boundary/non-manifold edges are silently skipped.
    ///
    /// @returns A vector of size NumFaces(), each entry being the list of
    ///          EdgeAdjInfo structs describing the neighbours of that triangle.
    std::vector<std::vector<EdgeAdjInfo>> BuildAdjacencyWithEdgeInfo() const {
        struct EdgeRec {
            size_t f;
            int a, b;
        };

        const size_t nf = faces.size();
        std::vector<std::vector<EdgeAdjInfo>> adj(nf);

        std::map<std::pair<int, int>, std::vector<EdgeRec>> edge_map;
        auto add_edge = [&](size_t f, int a, int b) {
            int lo = std::min(a, b), hi = std::max(a, b);
            edge_map[{lo, hi}].push_back(EdgeRec{f, a, b});
        };

        for (size_t i = 0; i < nf; ++i) {
            const auto& tri = faces[i];
            add_edge(i, tri[0], tri[1]);
            add_edge(i, tri[1], tri[2]);
            add_edge(i, tri[2], tri[0]);
        }

        for (const auto& kv : edge_map) {
            const auto& recs = kv.second;
            if (recs.size() != 2)
                continue;  // boundary or non-manifold — skip
            const EdgeRec& r0 = recs[0];
            const EdgeRec& r1 = recs[1];
            bool ok = (r0.a == r1.b && r0.b == r1.a);
            adj[r0.f].push_back(EdgeAdjInfo{r1.f, r0.a, r0.b, ok});
            adj[r1.f].push_back(EdgeAdjInfo{r0.f, r1.a, r1.b, ok});
        }

        return adj;
    }
};

}  // namespace mophi

#endif
