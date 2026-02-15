//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_MESH_HPP
#define MOPHI_MESH_HPP

#include <utils/robin_hood.h>
#include <common/VariableTypes.hpp>
#include <common/SharedStructs.hpp>
#include <core/DataClasses.hpp>
#include <core/Real3.hpp>
#include <kernels/Compression.cuh>

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

}  // namespace mophi

#endif
