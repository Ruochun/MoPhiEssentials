//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_MESH_IO_HPP
#define MOPHI_MESH_IO_HPP

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cassert>

#include "pugixml.hpp"
#include <common/SharedStructs.hpp>
#include <common/Mesh.hpp>

namespace mophi {

// ========================= VTU (XML) Loader =========================

inline const char* vtk_int_type_name_for_ids() {
    if constexpr (sizeof(nodeID_t) == 8)
        return "Int64";
    else
        return "Int32";
}

// === Base64 Decoder ===
inline std::vector<uint8_t> base64_decode(const std::string& input) {
    static constexpr unsigned char kDecTable[256] = {
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 64, 64, 64, 0,  64, 64, 64, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64, 64, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    std::vector<uint8_t> out;
    int val = 0, valb = -8;
    for (uint8_t c : input) {
        if (c > 255 || kDecTable[c] == 64)
            continue;
        val = (val << 6) + kDecTable[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// === DataArray Loader ===
template <typename T>
std::vector<T> ParseDataArray(const pugi::xml_node& da) {
    if (!da)
        MOPHI_ERROR(std::string("ParseDataArray: missing <DataArray> node."));

    const std::string format = da.attribute("format").as_string("ascii");
    std::vector<T> result;
    result.reserve(256);  // small default; grows as needed

    if (format == "ascii") {
        std::istringstream ss(da.child_value());
        ss.unsetf(std::ios::skipws);  // we'll manage whitespace explicitly
        // Re-enable whitespace skipping via >> by toggling it locally:
        ss >> std::skipws;

        if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
            // IMPORTANT: 8-bit ints stream as characters by default.
            // Parse into a wider integer, then cast.
            int tmp;
            while (ss >> tmp)
                result.push_back(static_cast<T>(tmp));
        } else if constexpr (std::is_same_v<T, bool>) {
            int tmp;
            while (ss >> tmp)
                result.push_back(tmp != 0);
        } else {
            T val;
            while (ss >> val)
                result.push_back(val);
        }
        return result;
    }

    if (format == "binary") {
        // VTU "binary" can be inline (base64) or appended; not supported here.
        MOPHI_ERROR(std::string("Binary VTU DataArray not supported yet (inline/appended)."));
    }

    MOPHI_ERROR("Unsupported DataArray format: " + format);
    return result;  // unreachable, keeps compiler happy
}

// === VTU Loader for partitioned Mesh (owned+halo) ===
// Requires: pugi XML + your ParseDataArray<T>(pugi::xml_node).
// Tolerates ASCII or appended binary DataArray.
// Invariants enforced on return: nodes are [0..nOwned) owned, [nOwned..N) halo.
inline Mesh LoadVtu(const std::string& filename) {
    Mesh mesh;

    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str()))
        MOPHI_ERROR("Failed to parse VTU file: " + filename);

    auto vtk = doc.child("VTKFile");
    auto piece = vtk.child("UnstructuredGrid").child("Piece");
    if (!piece)
        MOPHI_ERROR(std::string("Invalid VTU: missing UnstructuredGrid/Piece"));

    // ---- Points ----
    auto points_da = piece.child("Points").child("DataArray");
    auto coords = ParseDataArray<double>(points_da);
    if (coords.size() % 3 != 0)
        MOPHI_ERROR(std::string("VTU points not 3-component"));
    mesh.geom.nodes.resize(coords.size() / 3);
    for (size_t i = 0, j = 0; i < mesh.geom.nodes.size(); ++i) {
        mesh.geom.nodes[i] = Real3d{(double)coords[j++], (double)coords[j++], (double)coords[j++]};
    }

    // ---- Cells: conn/offset/types ----
    pugi::xml_node conn_da, offset_da, types_da;
    for (auto da : piece.child("Cells").children("DataArray")) {
        std::string name = da.attribute("Name").as_string();
        if (name == "connectivity")
            conn_da = da;
        else if (name == "offsets")
            offset_da = da;
        else if (name == "types")
            types_da = da;
    }
    if (!conn_da || !offset_da || !types_da)
        MOPHI_ERROR(std::string("VTU missing cells arrays"));

    auto connectivity = ParseDataArray<nodeID_t>(conn_da);
    auto offsets = ParseDataArray<nodeID_t>(offset_da);
    auto types = ParseDataArray<int>(types_da);

    // Optional CellData
    std::vector<nodeIDGlobal_t> globalCellIDs;  // length = (#tets + #hexes + (maybe triangles))
    std::vector<int> regionTags;                // same length
    if (auto cellData = piece.child("CellData")) {
        for (auto da : cellData.children("DataArray")) {
            std::string name = da.attribute("Name").as_string();
            if (name == "GlobalCellID") {
                globalCellIDs = ParseDataArray<nodeIDGlobal_t>(da);
            } else if (name == "CellRegionTag") {
                regionTags = ParseDataArray<int>(da);
            }
        }
    }

    // Optional FieldData (PartID, NumOwnedNodes)
    int fieldPartID = -1;
    nodeID_t fieldNumOwnedNodes = -1;
    if (auto field = vtk.child("FieldData")) {
        for (auto da : field.children("DataArray")) {
            std::string name = da.attribute("Name").as_string();
            if (name == "PartID") {
                auto v = ParseDataArray<int>(da);
                if (!v.empty())
                    fieldPartID = v[0];
            } else if (name == "NumOwnedNodes") {
                auto v = ParseDataArray<nodeID_t>(da);
                if (!v.empty())
                    fieldNumOwnedNodes = v[0];
            }
        }
    }

    // Optional PointData (GlobalNodeID, NodeIsOwned)
    std::vector<nodeIDGlobal_t> globalNodeIDs;
    std::vector<uint8_t> nodeIsOwned;  // 1 = owned, 0 = halo
    if (auto pd = piece.child("PointData")) {
        for (auto da : pd.children("DataArray")) {
            std::string name = da.attribute("Name").as_string();
            if (name == "GlobalNodeID") {
                globalNodeIDs = ParseDataArray<nodeIDGlobal_t>(da);
            } else if (name == "NodeIsOwned") {
                // Could be UInt8 written as Int32/UInt8; accept both
                auto type = std::string(da.attribute("type").as_string());
                if (type == "UInt8")
                    nodeIsOwned = ParseDataArray<uint8_t>(da);
                else {
                    auto tmp = ParseDataArray<int>(da);
                    nodeIsOwned.resize(tmp.size());
                    for (size_t i = 0; i < tmp.size(); ++i)
                        nodeIsOwned[i] = (uint8_t)tmp[i];
                }
            }
        }
    }

    // ---- Build topology: Tets (type 10 tet4, type 24 tet10) & Hexes (type 12); ignore lower-dim elements ----
    size_t start = 0;
    size_t tetCount = 0, hexCount = 0;
    mesh.topo.tets.clear();
    mesh.topo.hexes.clear();

    // Reserve (rough heuristic) for fewer reallocs
    mesh.topo.tets.reserve(types.size());
    mesh.topo.hexes.reserve(types.size());

    auto read_tag = [&](size_t cellIndex) -> int {
        if (regionTags.empty())
            return 0;
        if (cellIndex < regionTags.size())
            return regionTags[cellIndex];
        return 0;
    };
    auto read_global_cell = [&](size_t cellIndex) -> nodeIDGlobal_t {
        if (globalCellIDs.empty())
            return (nodeIDGlobal_t)cellIndex;
        if (cellIndex < globalCellIDs.size())
            return globalCellIDs[cellIndex];
        return (nodeIDGlobal_t)cellIndex;
    };

    mesh.localToGlobalCell.clear();
    mesh.localToGlobalCell.reserve(types.size());

    for (size_t i = 0; i < types.size(); ++i) {
        size_t end = (size_t)offsets[i];
        if (end > connectivity.size() || end < start)
            MOPHI_ERROR(std::string("Malformed VTU offsets/connectivity"));

        int cell_type = types[i];
        const size_t nverts = end - start;

        if ((cell_type == 10 && nverts == 4) || (cell_type == 24 && nverts == 10)) {
            // Linear tetrahedron (tet4) or Quadratic tetrahedron (tet10)
            // For tet10: extract corner nodes only (VTK ordering: 0-3 are corners, 4-9 are midpoints)
            std::array<nodeID_t, 4> v{connectivity[start + 0], connectivity[start + 1], connectivity[start + 2],
                                      connectivity[start + 3]};
            mesh.topo.tets.push_back(v);
            mesh.topo.tetTags.push_back(read_tag(i));
            mesh.localToGlobalCell.push_back(read_global_cell(i));
            tetCount++;
        } else if (cell_type == 12 && nverts == 8) {
            // Linear hexahedron
            std::array<nodeID_t, 8> v{connectivity[start + 0], connectivity[start + 1], connectivity[start + 2],
                                      connectivity[start + 3], connectivity[start + 4], connectivity[start + 5],
                                      connectivity[start + 6], connectivity[start + 7]};
            mesh.topo.hexes.push_back(v);
            mesh.topo.hexTags.push_back(read_tag(i));
            mesh.localToGlobalCell.push_back(read_global_cell(i));
            hexCount++;
        } else if (cell_type == 5 || cell_type == 9 || cell_type == 3 || cell_type == 1) {
            // Silently ignore lower-dimensional elements (triangles, quads, lines, vertices)
        } else {
            // Error on unsupported 3D cell types
            std::ostringstream msg;
            msg << "Unsupported VTU cell type " << cell_type << " with " << nverts << " vertices at cell index " << i
                << ". Supported types: tet4 (type=10, nverts=4), tet10 (type=24, nverts=10), hex8 (type=12, nverts=8).";
            MOPHI_ERROR(msg.str());
        }
        start = end;
    }

    // ---- Node ownership / global IDs / halo maps ----
    mesh.partID = fieldPartID;

    // Global node IDs
    const size_t N = mesh.geom.nodes.size();
    mesh.halo.localToGlobalNode.resize(N);
    if (!globalNodeIDs.empty() && globalNodeIDs.size() == N) {
        mesh.halo.localToGlobalNode = globalNodeIDs;
    } else {
        // Fallback: identity
        for (nodeID_t i = 0; i < (nodeID_t)N; ++i)
            mesh.halo.localToGlobalNode[i] = (nodeIDGlobal_t)i;
    }

    // Owned count
    nodeID_t ownedCount = (fieldNumOwnedNodes >= 0) ? (nodeID_t)fieldNumOwnedNodes
                          : !nodeIsOwned.empty()
                              ? (nodeID_t)std::count(nodeIsOwned.begin(), nodeIsOwned.end(), (uint8_t)1)
                              : (nodeID_t)N;  // default: all owned
    mesh.geom.nOwnedNodes = ownedCount;

    // If NodeIsOwned exists but owned nodes aren’t already [0..ownedCount), reorder
    bool need_reorder = false;
    if (!nodeIsOwned.empty()) {
        for (nodeID_t i = 0; i < ownedCount; ++i)
            if (nodeIsOwned[i] == 0) {
                need_reorder = true;
                break;
            }
        for (nodeID_t i = ownedCount; i < (nodeID_t)N && !need_reorder; ++i)
            if (nodeIsOwned[i] == 1) {
                need_reorder = true;
                break;
            }
    }

    if (need_reorder) {
        // Build permutation: owned first (ascending old index), then halo
        std::vector<nodeID_t> ownedIdx, haloIdx;
        ownedIdx.reserve(ownedCount);
        haloIdx.reserve(N - ownedCount);
        for (nodeID_t i = 0; i < (nodeID_t)N; ++i) {
            if (nodeIsOwned[i])
                ownedIdx.push_back(i);
            else
                haloIdx.push_back(i);
        }
        if (ownedIdx.size() != ownedCount)
            MOPHI_ERROR(std::string("NodeIsOwned inconsistency"));
        std::vector<nodeID_t> permOldToNew(N);
        nodeID_t p = 0;
        for (auto i : ownedIdx)
            permOldToNew[i] = p++;
        for (auto i : haloIdx)
            permOldToNew[i] = p++;

        // Apply permutation to nodes
        std::vector<Real3d> newNodes(N);
        for (nodeID_t old = 0; old < (nodeID_t)N; ++old)
            newNodes[permOldToNew[old]] = mesh.geom.nodes[old];
        mesh.geom.nodes.swap(newNodes);

        // Remap connectivity
        for (auto& t : mesh.topo.tets)
            for (int k = 0; k < 4; ++k)
                t[k] = permOldToNew[t[k]];
        for (auto& h : mesh.topo.hexes)
            for (int k = 0; k < 8; ++k)
                h[k] = permOldToNew[h[k]];

        // Remap localToGlobalNode
        std::vector<nodeIDGlobal_t> newL2G(N);
        for (nodeID_t old = 0; old < (nodeID_t)N; ++old)
            newL2G[permOldToNew[old]] = mesh.halo.localToGlobalNode[old];
        mesh.halo.localToGlobalNode.swap(newL2G);
    }

    // Build globalToLocal map
    mesh.halo.globalToLocalNode.clear();
    mesh.halo.globalToLocalNode.reserve(mesh.halo.localToGlobalNode.size() * 1.3);
    for (nodeID_t l = 0; l < (nodeID_t)mesh.halo.localToGlobalNode.size(); ++l) {
        mesh.halo.globalToLocalNode.emplace(mesh.halo.localToGlobalNode[l], l);
    }

    // Ensure tags vector sizes match cell counts
    if (mesh.topo.tetTags.size() != mesh.topo.tets.size())
        mesh.topo.tetTags.assign(mesh.topo.tets.size(), 0);
    if (mesh.topo.hexTags.size() != mesh.topo.hexes.size())
        mesh.topo.hexTags.assign(mesh.topo.hexes.size(), 0);

    return mesh;
}

// === VTU Writer for partitioned Mesh (owned+halo) ===
// Writes ASCII VTU with PointData/CellData carrying globals & ownership.
// Cell order = all tets, then all hexes. Tags align with that order.
inline void WriteVtu(const std::string& filename, const Mesh& mesh) {
    pugi::xml_document doc;

    auto vtk = doc.append_child("VTKFile");
    vtk.append_attribute("type") = "UnstructuredGrid";
    vtk.append_attribute("version") = "1.0";
    vtk.append_attribute("byte_order") = "LittleEndian";

    // Optional FieldData (metadata)
    {
        auto fd = vtk.append_child("FieldData");
        auto daPart = fd.append_child("DataArray");
        daPart.append_attribute("type") = "Int32";
        daPart.append_attribute("Name") = "PartID";
        daPart.append_attribute("NumberOfTuples") = 1;
        daPart.append_attribute("format") = "ascii";
        daPart.append_child(pugi::node_pcdata).set_value(std::to_string(mesh.partID).c_str());

        auto daOwned = fd.append_child("DataArray");
        daOwned.append_attribute("type") = "Int32";
        daOwned.append_attribute("Name") = "NumOwnedNodes";
        daOwned.append_attribute("NumberOfTuples") = 1;
        daOwned.append_attribute("format") = "ascii";
        daOwned.append_child(pugi::node_pcdata).set_value(std::to_string((int)mesh.geom.nOwnedNodes).c_str());
    }

    auto grid = vtk.append_child("UnstructuredGrid");
    auto piece = grid.append_child("Piece");
    const nodeID_t nPoints = (nodeID_t)mesh.geom.nodes.size();
    const nodeID_t nCells = (nodeID_t)(mesh.topo.tets.size() + mesh.topo.hexes.size());
    piece.append_attribute("NumberOfPoints") = nPoints;
    piece.append_attribute("NumberOfCells") = nCells;

    // Points
    {
        auto points = piece.append_child("Points").append_child("DataArray");
        points.append_attribute("type") = "Float64";
        points.append_attribute("NumberOfComponents") = "3";
        points.append_attribute("format") = "ascii";
        std::ostringstream pbuf;
        pbuf.setf(std::ios::scientific);
        pbuf.precision(17);
        for (const auto& p : mesh.geom.nodes)
            pbuf << (double)p.x() << " " << (double)p.y() << " " << (double)p.z() << "\n";
        points.append_child(pugi::node_pcdata).set_value(pbuf.str().c_str());
    }

    // Cells
    {
        auto cells = piece.append_child("Cells");

        // connectivity
        auto conn = cells.append_child("DataArray");
        conn.append_attribute("type") = vtk_int_type_name_for_ids();
        conn.append_attribute("Name") = "connectivity";
        conn.append_attribute("format") = "ascii";
        std::ostringstream cbuf;
        for (const auto& t : mesh.topo.tets)
            cbuf << t[0] << " " << t[1] << " " << t[2] << " " << t[3] << " ";
        for (const auto& h : mesh.topo.hexes)
            for (int k = 0; k < 8; ++k)
                cbuf << h[k] << " ";
        conn.append_child(pugi::node_pcdata).set_value(cbuf.str().c_str());

        // offsets
        auto offs = cells.append_child("DataArray");
        offs.append_attribute("type") = vtk_int_type_name_for_ids();
        offs.append_attribute("Name") = "offsets";
        offs.append_attribute("format") = "ascii";
        std::ostringstream obuf;
        nodeID_t off = 0;
        for (size_t i = 0; i < mesh.topo.tets.size(); ++i) {
            off += 4;
            obuf << off << " ";
        }
        for (size_t i = 0; i < mesh.topo.hexes.size(); ++i) {
            off += 8;
            obuf << off << " ";
        }
        offs.append_child(pugi::node_pcdata).set_value(obuf.str().c_str());

        // types
        auto types = cells.append_child("DataArray");
        types.append_attribute("type") = "UInt8";
        types.append_attribute("Name") = "types";
        types.append_attribute("format") = "ascii";
        std::ostringstream tbuf;
        for (size_t i = 0; i < mesh.topo.tets.size(); ++i)
            tbuf << "10 ";
        for (size_t i = 0; i < mesh.topo.hexes.size(); ++i)
            tbuf << "12 ";
        types.append_child(pugi::node_pcdata).set_value(tbuf.str().c_str());
    }

    // PointData: GlobalNodeID, NodeIsOwned
    {
        auto pd = piece.append_child("PointData");

        // GlobalNodeID
        auto gni = pd.append_child("DataArray");
        gni.append_attribute("type") = (sizeof(nodeIDGlobal_t) == 8) ? "Int64" : "Int32";
        gni.append_attribute("Name") = "GlobalNodeID";
        gni.append_attribute("format") = "ascii";
        std::ostringstream gbuf;
        for (auto g : mesh.halo.localToGlobalNode)
            gbuf << g << " ";
        pd.last_child().append_child(pugi::node_pcdata).set_value(gbuf.str().c_str());

        // NodeIsOwned
        auto nio = pd.append_child("DataArray");
        nio.append_attribute("type") = "UInt8";
        nio.append_attribute("Name") = "NodeIsOwned";
        nio.append_attribute("format") = "ascii";
        std::ostringstream obuf;
        for (nodeID_t i = 0; i < (nodeID_t)mesh.geom.nodes.size(); ++i)
            obuf << ((i < mesh.geom.nOwnedNodes) ? 1 : 0) << " ";
        pd.last_child().append_child(pugi::node_pcdata).set_value(obuf.str().c_str());
    }

    // CellData: GlobalCellID, CellRegionTag (aligned: tets then hexes)
    {
        auto cd = piece.append_child("CellData");
        const nodeID_t nCells = (nodeID_t)(mesh.topo.tets.size() + mesh.topo.hexes.size());

        // GlobalCellID (optional)
        if (!mesh.localToGlobalCell.empty() && mesh.localToGlobalCell.size() == (size_t)nCells) {
            auto gci = cd.append_child("DataArray");
            gci.append_attribute("type") = (sizeof(nodeID_t) == 8) ? "Int64" : "Int32";
            gci.append_attribute("Name") = "GlobalCellID";
            gci.append_attribute("format") = "ascii";
            gci.append_attribute("NumberOfTuples") = (unsigned)nCells;

            std::ostringstream gbuf;
            for (auto g : mesh.localToGlobalCell)
                gbuf << g << " ";
            gci.append_child(pugi::node_pcdata).set_value(gbuf.str().c_str());
        }

        // CellRegionTag
        auto crt = cd.append_child("DataArray");
        crt.append_attribute("type") = "Int32";
        crt.append_attribute("Name") = "CellRegionTag";
        crt.append_attribute("format") = "ascii";
        crt.append_attribute("NumberOfTuples") = (unsigned)nCells;

        std::ostringstream rbuf;
        // tets first
        if (mesh.topo.tetTags.size() == mesh.topo.tets.size()) {
            for (int v : mesh.topo.tetTags)
                rbuf << v << " ";
        } else {
            for (size_t i = 0; i < mesh.topo.tets.size(); ++i)
                rbuf << 0 << " ";
        }
        // hexes next
        if (mesh.topo.hexTags.size() == mesh.topo.hexes.size()) {
            for (int v : mesh.topo.hexTags)
                rbuf << v << " ";
        } else {
            for (size_t i = 0; i < mesh.topo.hexes.size(); ++i)
                rbuf << 0 << " ";
        }
        crt.append_child(pugi::node_pcdata).set_value(rbuf.str().c_str());
    }

    if (!doc.save_file(filename.c_str()))
        MOPHI_ERROR("Failed to write VTU file: " + filename);
}

}  // namespace mophi

#endif
