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
#include <cstdint>
#include <cstring>
#include <cmath>
#include <map>

#include "pugixml.hpp"
#include "../common/SharedStructs.hpp"
#include "../common/Mesh.hpp"
#include "../core/WavefrontMeshLoader.hpp"

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

    // ---- Build topology: only Tets (VTK type 10) & Tet10s (VTK type 24) & Hexes (VTK type 12); ignore Triangles (VTK
    // type 5) ----
    size_t start = 0;
    size_t tetCount = 0, tet10Count = 0, hexCount = 0;
    mesh.topo.tets.clear();
    mesh.topo.tet10s.clear();
    mesh.topo.hexes.clear();

    // Reserve (rough heuristic) for fewer reallocs
    mesh.topo.tets.reserve(types.size());
    mesh.topo.tet10s.reserve(types.size());
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

        if (cell_type == 10 && nverts == 4) {
            std::array<nodeID_t, 4> v{connectivity[start + 0], connectivity[start + 1], connectivity[start + 2],
                                      connectivity[start + 3]};
            mesh.topo.tets.push_back(v);
            mesh.topo.tetTags.push_back(read_tag(i));
            mesh.localToGlobalCell.push_back(read_global_cell(i));
            tetCount++;
        } else if (cell_type == 24 && nverts == 10) {
            std::array<nodeID_t, 10> v{connectivity[start + 0], connectivity[start + 1], connectivity[start + 2],
                                       connectivity[start + 3], connectivity[start + 4], connectivity[start + 5],
                                       connectivity[start + 6], connectivity[start + 7], connectivity[start + 8],
                                       connectivity[start + 9]};
            mesh.topo.tet10s.push_back(v);
            mesh.topo.tet10Tags.push_back(read_tag(i));
            mesh.localToGlobalCell.push_back(read_global_cell(i));
            tet10Count++;
        } else if (cell_type == 12 && nverts == 8) {
            std::array<nodeID_t, 8> v{connectivity[start + 0], connectivity[start + 1], connectivity[start + 2],
                                      connectivity[start + 3], connectivity[start + 4], connectivity[start + 5],
                                      connectivity[start + 6], connectivity[start + 7]};
            mesh.topo.hexes.push_back(v);
            mesh.topo.hexTags.push_back(read_tag(i));
            mesh.localToGlobalCell.push_back(read_global_cell(i));
            hexCount++;
        } else {
            // Ignore other types (e.g., triangles=5)
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
        for (auto& t10 : mesh.topo.tet10s)
            for (int k = 0; k < 10; ++k)
                t10[k] = permOldToNew[t10[k]];
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
    if (mesh.topo.tet10Tags.size() != mesh.topo.tet10s.size())
        mesh.topo.tet10Tags.assign(mesh.topo.tet10s.size(), 0);
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
    const nodeID_t nCells = (nodeID_t)(mesh.topo.tets.size() + mesh.topo.tet10s.size() + mesh.topo.hexes.size());
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
        for (const auto& t10 : mesh.topo.tet10s)
            for (int k = 0; k < 10; ++k)
                cbuf << t10[k] << " ";
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
        for (size_t i = 0; i < mesh.topo.tet10s.size(); ++i) {
            off += 10;
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
        for (size_t i = 0; i < mesh.topo.tet10s.size(); ++i)
            tbuf << "24 ";
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

    // CellData: GlobalCellID, CellRegionTag (aligned: tets then tet10s then hexes)
    {
        auto cd = piece.append_child("CellData");
        const nodeID_t nCells = (nodeID_t)(mesh.topo.tets.size() + mesh.topo.tet10s.size() + mesh.topo.hexes.size());

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
        // tet10s next
        if (mesh.topo.tet10Tags.size() == mesh.topo.tet10s.size()) {
            for (int v : mesh.topo.tet10Tags)
                rbuf << v << " ";
        } else {
            for (size_t i = 0; i < mesh.topo.tet10s.size(); ++i)
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

// =============================================================================
// SurfaceMesh STL I/O
// =============================================================================

/// Load a surface mesh from an STL file (binary or ASCII auto-detected).
///
/// @param filename     Path to the .stl file.
/// @param mesh         Output mesh (cleared first).
/// @param load_normals If true, compute per-face geometric normals from vertex
///                     winding (the STL file's embedded normals are ignored as
///                     they can be inconsistent).
/// @returns true on success.
inline bool LoadSTL(const std::string& filename, SurfaceMesh& mesh, bool load_normals = false) {
    mesh.Clear();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "LoadSTL: cannot open file: " << filename << "\n";
        return false;
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (buffer.size() < 84) {
        std::cerr << "LoadSTL: file too small: " << filename << "\n";
        return false;
    }

    // Heuristic: binary STL begins with an 80-byte header followed by a 4-byte
    // triangle count, then exactly (count * 50) bytes of triangle data.
    uint32_t tri_count = 0;
    std::memcpy(&tri_count, buffer.data() + 80, sizeof(uint32_t));
    const size_t expected_bin = 84 + static_cast<size_t>(tri_count) * 50;
    const bool looks_binary = (expected_bin == buffer.size());

    bool parsed = false;

    if (looks_binary) {
        const unsigned char* data = reinterpret_cast<const unsigned char*>(buffer.data());
        size_t offset = 84;
        for (uint32_t i = 0; i < tri_count; ++i) {
            float floats[12];
            std::memcpy(floats, data + offset, sizeof(float) * 12);
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back(Real3d(floats[3], floats[4], floats[5]));
            mesh.vertices.push_back(Real3d(floats[6], floats[7], floats[8]));
            mesh.vertices.push_back(Real3d(floats[9], floats[10], floats[11]));
            mesh.faces.push_back({base, base + 1, base + 2});
            offset += 50;
        }
        parsed = !mesh.faces.empty();
    }

    if (!parsed) {
        // Fallback: ASCII STL
        std::istringstream iss(std::string(buffer.begin(), buffer.end()));
        std::string line;
        std::vector<Real3d> facet_verts;
        facet_verts.reserve(3);
        while (std::getline(iss, line)) {
            std::istringstream ls(line);
            std::string token;
            ls >> token;
            if (token == "facet") {
                facet_verts.clear();
            } else if (token == "vertex") {
                double vx, vy, vz;
                if (ls >> vx >> vy >> vz) {
                    facet_verts.push_back(Real3d(vx, vy, vz));
                    if (facet_verts.size() == 3) {
                        const int base = static_cast<int>(mesh.vertices.size());
                        mesh.vertices.push_back(facet_verts[0]);
                        mesh.vertices.push_back(facet_verts[1]);
                        mesh.vertices.push_back(facet_verts[2]);
                        mesh.faces.push_back({base, base + 1, base + 2});
                    }
                }
            }
        }
        parsed = !mesh.faces.empty();
    }

    if (!parsed || mesh.faces.empty()) {
        std::cerr << "LoadSTL: failed to parse: " << filename << "\n";
        return false;
    }

    if (load_normals)
        mesh.ComputeFaceNormals();

    return true;
}

/// Write a surface mesh to an STL file.
///
/// @param filename  Path to the output .stl file.
/// @param mesh      Mesh to write.
/// @param binary    If true (default), write binary STL; otherwise write ASCII.
/// @returns true on success.
inline bool WriteSTL(const std::string& filename, const SurfaceMesh& mesh, bool binary = true) {
    if (binary) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "WriteSTL: cannot open file for writing: " << filename << "\n";
            return false;
        }

        // 80-byte header
        char header[80] = {};
        const std::string hdr = "Binary STL written by MoPhiEssentials";
        std::copy_n(hdr.begin(), std::min(hdr.size(), static_cast<size_t>(79)), header);
        file.write(header, 80);

        const uint32_t ntris = static_cast<uint32_t>(mesh.faces.size());
        file.write(reinterpret_cast<const char*>(&ntris), sizeof(uint32_t));

        const uint16_t attr = 0;
        for (size_t i = 0; i < mesh.faces.size(); ++i) {
            const auto& f = mesh.faces[i];
            const Real3d& v0 = mesh.vertices[(size_t)f[0]];
            const Real3d& v1 = mesh.vertices[(size_t)f[1]];
            const Real3d& v2 = mesh.vertices[(size_t)f[2]];

            Real3d e1 = v1 - v0;
            Real3d e2 = v2 - v0;
            Real3d n = e1 % e2;
            const double len = n.Length();
            if (len > kMeshNormalLengthEps)
                n = n * (1.0 / len);

            float nf[3] = {static_cast<float>(n.x()), static_cast<float>(n.y()), static_cast<float>(n.z())};
            file.write(reinterpret_cast<const char*>(nf), sizeof(float) * 3);

            float vf[3];
            vf[0] = static_cast<float>(v0.x());
            vf[1] = static_cast<float>(v0.y());
            vf[2] = static_cast<float>(v0.z());
            file.write(reinterpret_cast<const char*>(vf), sizeof(float) * 3);
            vf[0] = static_cast<float>(v1.x());
            vf[1] = static_cast<float>(v1.y());
            vf[2] = static_cast<float>(v1.z());
            file.write(reinterpret_cast<const char*>(vf), sizeof(float) * 3);
            vf[0] = static_cast<float>(v2.x());
            vf[1] = static_cast<float>(v2.y());
            vf[2] = static_cast<float>(v2.z());
            file.write(reinterpret_cast<const char*>(vf), sizeof(float) * 3);

            file.write(reinterpret_cast<const char*>(&attr), sizeof(uint16_t));
        }
        return static_cast<bool>(file);
    } else {
        // ASCII STL
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "WriteSTL: cannot open file for writing: " << filename << "\n";
            return false;
        }
        file << std::scientific;
        file.precision(8);
        file << "solid mesh\n";
        for (size_t i = 0; i < mesh.faces.size(); ++i) {
            const auto& f = mesh.faces[i];
            const Real3d& v0 = mesh.vertices[(size_t)f[0]];
            const Real3d& v1 = mesh.vertices[(size_t)f[1]];
            const Real3d& v2 = mesh.vertices[(size_t)f[2]];

            Real3d e1 = v1 - v0;
            Real3d e2 = v2 - v0;
            Real3d n = e1 % e2;
            const double len = n.Length();
            if (len > kMeshNormalLengthEps)
                n = n * (1.0 / len);

            file << "facet normal " << n.x() << " " << n.y() << " " << n.z() << "\n";
            file << "  outer loop\n";
            file << "    vertex " << v0.x() << " " << v0.y() << " " << v0.z() << "\n";
            file << "    vertex " << v1.x() << " " << v1.y() << " " << v1.z() << "\n";
            file << "    vertex " << v2.x() << " " << v2.y() << " " << v2.z() << "\n";
            file << "  endloop\n";
            file << "endfacet\n";
        }
        file << "endsolid mesh\n";
        return static_cast<bool>(file);
    }
}

// =============================================================================
// SurfaceMesh PLY I/O
// =============================================================================

/// Load a surface mesh from a PLY file (ASCII or binary little-endian).
///
/// Vertex properties x, y, z are required; nx, ny, nz are optional.
/// Big-endian binary PLY is not supported.
///
/// @param filename     Path to the .ply file.
/// @param mesh         Output mesh (cleared first).
/// @param load_normals If true, compute per-face geometric normals from vertex
///                     winding (overwriting any per-vertex normals stored in
///                     the file).
/// @returns true on success.
inline bool LoadPLY(const std::string& filename, SurfaceMesh& mesh, bool load_normals = false) {
    mesh.Clear();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "LoadPLY: cannot open file: " << filename << "\n";
        return false;
    }

    // --- Parse header ---
    std::string line;
    if (!std::getline(file, line) || line.rfind("ply", 0) != 0) {
        std::cerr << "LoadPLY: missing 'ply' magic: " << filename << "\n";
        return false;
    }

    enum class PLYFormat { ASCII, BINARY_LE, BINARY_BE };
    PLYFormat fmt = PLYFormat::ASCII;
    size_t num_vertices = 0, num_faces = 0;
    std::vector<std::string> vprop_names, vprop_types;
    bool in_vertex = false;
    std::string face_count_type, face_index_type;

    while (std::getline(file, line)) {
        if (line == "end_header")
            break;
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "format") {
            std::string f;
            ls >> f;
            if (f.find("ascii") == 0)
                fmt = PLYFormat::ASCII;
            else if (f.find("binary_little_endian") == 0)
                fmt = PLYFormat::BINARY_LE;
            else if (f.find("binary_big_endian") == 0)
                fmt = PLYFormat::BINARY_BE;
        } else if (token == "element") {
            std::string elem;
            ls >> elem;
            if (elem == "vertex") {
                ls >> num_vertices;
                in_vertex = true;
            } else if (elem == "face") {
                ls >> num_faces;
                in_vertex = false;
            } else {
                in_vertex = false;
            }
        } else if (token == "property" && in_vertex) {
            std::string type, name;
            ls >> type >> name;
            if (!name.empty()) {
                vprop_names.push_back(name);
                vprop_types.push_back(type);
            }
        } else if (token == "property" && !in_vertex) {
            std::string maybe_list;
            ls >> maybe_list;
            if (maybe_list == "list") {
                ls >> face_count_type >> face_index_type;
            }
        }
    }

    if (fmt == PLYFormat::BINARY_BE) {
        std::cerr << "LoadPLY: big-endian binary PLY not supported: " << filename << "\n";
        return false;
    }
    if (num_vertices == 0 || num_faces == 0) {
        std::cerr << "LoadPLY: no vertices or faces: " << filename << "\n";
        return false;
    }

    auto find_prop = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(vprop_names.size()); ++i)
            if (vprop_names[i] == name)
                return i;
        return -1;
    };
    const int idx_x = find_prop("x"), idx_y = find_prop("y"), idx_z = find_prop("z");
    const int idx_nx = find_prop("nx"), idx_ny = find_prop("ny"), idx_nz = find_prop("nz");
    const bool has_vnormals = (idx_nx >= 0 && idx_ny >= 0 && idx_nz >= 0);

    // Helper: read a single scalar of the given PLY type (binary LE)
    auto read_scalar_le = [&](const std::string& type, double& out) -> bool {
        if (type == "float" || type == "float32") {
            float v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(float)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "double" || type == "float64") {
            double v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(double)))
                return false;
            out = v;
        } else if (type == "uchar" || type == "uint8") {
            uint8_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(uint8_t)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "char" || type == "int8") {
            int8_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(int8_t)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "short" || type == "int16") {
            int16_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(int16_t)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "ushort" || type == "uint16") {
            uint16_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(uint16_t)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "int" || type == "int32") {
            int32_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(int32_t)))
                return false;
            out = static_cast<double>(v);
        } else if (type == "uint" || type == "uint32") {
            uint32_t v;
            if (!file.read(reinterpret_cast<char*>(&v), sizeof(uint32_t)))
                return false;
            out = static_cast<double>(v);
        } else {
            return false;
        }
        return true;
    };

    mesh.vertices.reserve(num_vertices);
    mesh.faces.reserve(num_faces);
    std::vector<Real3d> file_normals;

    // --- Read vertices ---
    for (size_t i = 0; i < num_vertices; ++i) {
        if (fmt == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                std::cerr << "LoadPLY: unexpected EOF in vertices: " << filename << "\n";
                return false;
            }
            std::istringstream ls(line);
            std::vector<double> vals;
            double v;
            while (ls >> v)
                vals.push_back(v);

            if (idx_x < 0 || idx_y < 0 || idx_z < 0 ||
                vals.size() <= static_cast<size_t>(std::max({idx_x, idx_y, idx_z}))) {
                std::cerr << "LoadPLY: missing xyz in vertex: " << filename << "\n";
                return false;
            }
            mesh.vertices.push_back(Real3d(vals[(size_t)idx_x], vals[(size_t)idx_y], vals[(size_t)idx_z]));
            if (has_vnormals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                file_normals.push_back(Real3d(vals[(size_t)idx_nx], vals[(size_t)idx_ny], vals[(size_t)idx_nz]));
            }
        } else {
            std::vector<double> vals(vprop_names.size(), 0.0);
            for (size_t p = 0; p < vprop_names.size(); ++p) {
                if (!read_scalar_le(vprop_types[p], vals[p])) {
                    std::cerr << "LoadPLY: error reading binary vertex: " << filename << "\n";
                    return false;
                }
            }
            if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
                std::cerr << "LoadPLY: missing xyz in vertex: " << filename << "\n";
                return false;
            }
            mesh.vertices.push_back(Real3d(vals[(size_t)idx_x], vals[(size_t)idx_y], vals[(size_t)idx_z]));
            if (has_vnormals && vals.size() > static_cast<size_t>(std::max({idx_nx, idx_ny, idx_nz}))) {
                file_normals.push_back(Real3d(vals[(size_t)idx_nx], vals[(size_t)idx_ny], vals[(size_t)idx_nz]));
            }
        }
    }

    // --- Read faces ---
    for (size_t i = 0; i < num_faces; ++i) {
        if (fmt == PLYFormat::ASCII) {
            if (!std::getline(file, line)) {
                std::cerr << "LoadPLY: unexpected EOF in faces: " << filename << "\n";
                return false;
            }
            std::istringstream ls(line);
            int verts_in_face = 0;
            ls >> verts_in_face;
            if (verts_in_face < 3)
                continue;
            std::vector<int> idx((size_t)verts_in_face);
            for (int j = 0; j < verts_in_face; ++j)
                ls >> idx[(size_t)j];
            // Fan-triangulate polygon
            for (int t = 1; t < verts_in_face - 1; ++t)
                mesh.faces.push_back({idx[0], idx[(size_t)t], idx[(size_t)t + 1]});
        } else {
            const std::string& cnt_type = face_count_type.empty() ? std::string("uchar") : face_count_type;
            const std::string& idx_type = face_index_type.empty() ? std::string("int") : face_index_type;
            double count_d = 0.0;
            if (!read_scalar_le(cnt_type, count_d)) {
                std::cerr << "LoadPLY: error reading face count: " << filename << "\n";
                return false;
            }
            const int verts_in_face = static_cast<int>(count_d);
            std::vector<int> idx((size_t)std::max(verts_in_face, 0));
            for (int j = 0; j < verts_in_face; ++j) {
                double v = 0.0;
                if (!read_scalar_le(idx_type, v)) {
                    std::cerr << "LoadPLY: error reading face indices: " << filename << "\n";
                    return false;
                }
                idx[(size_t)j] = static_cast<int>(v);
            }
            if (verts_in_face >= 3) {
                for (int t = 1; t < verts_in_face - 1; ++t)
                    mesh.faces.push_back({idx[0], idx[(size_t)t], idx[(size_t)t + 1]});
            }
        }
    }

    if (mesh.faces.empty()) {
        std::cerr << "LoadPLY: no faces parsed: " << filename << "\n";
        return false;
    }

    if (load_normals) {
        mesh.ComputeFaceNormals();
    } else if (!file_normals.empty() && file_normals.size() == mesh.vertices.size()) {
        // Store per-vertex normals from the file if available
        mesh.normals = std::move(file_normals);
        mesh.faceNormalIndices.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces)
            mesh.faceNormalIndices.push_back({f[0], f[1], f[2]});
    }

    return true;
}

/// Write a surface mesh to an ASCII or binary little-endian PLY file.
///
/// @param filename  Path to the output .ply file.
/// @param mesh      Mesh to write.
/// @param binary    If true, write binary little-endian PLY; otherwise ASCII.
/// @returns true on success.
inline bool WritePLY(const std::string& filename, const SurfaceMesh& mesh, bool binary = false) {
    if (binary) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "WritePLY: cannot open file for writing: " << filename << "\n";
            return false;
        }

        // Write header as text
        std::ostringstream hdr;
        hdr << "ply\n";
        hdr << "format binary_little_endian 1.0\n";
        hdr << "element vertex " << mesh.vertices.size() << "\n";
        hdr << "property float x\n";
        hdr << "property float y\n";
        hdr << "property float z\n";
        hdr << "element face " << mesh.faces.size() << "\n";
        hdr << "property list uchar int vertex_indices\n";
        hdr << "end_header\n";
        const std::string hdr_str = hdr.str();
        file.write(hdr_str.c_str(), (std::streamsize)hdr_str.size());

        for (const auto& v : mesh.vertices) {
            float x = static_cast<float>(v.x());
            float y = static_cast<float>(v.y());
            float z = static_cast<float>(v.z());
            file.write(reinterpret_cast<const char*>(&x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&z), sizeof(float));
        }

        const uint8_t three = 3;
        for (const auto& f : mesh.faces) {
            file.write(reinterpret_cast<const char*>(&three), sizeof(uint8_t));
            const int32_t ia = f[0], ib = f[1], ic = f[2];
            file.write(reinterpret_cast<const char*>(&ia), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&ib), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&ic), sizeof(int32_t));
        }
        return static_cast<bool>(file);
    } else {
        // ASCII PLY
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "WritePLY: cannot open file for writing: " << filename << "\n";
            return false;
        }

        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "element vertex " << mesh.vertices.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "element face " << mesh.faces.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";

        file << std::scientific;
        file.precision(8);
        for (const auto& v : mesh.vertices)
            file << v.x() << " " << v.y() << " " << v.z() << "\n";
        for (const auto& f : mesh.faces)
            file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

        return static_cast<bool>(file);
    }
}

// =============================================================================
// SurfaceMesh OBJ I/O
// =============================================================================

/// Load a surface mesh from a Wavefront OBJ file.
///
/// Uses the bundled WavefrontMeshLoader. Vertex, normal, and UV index pools are
/// stored verbatim from the file (0-based after conversion from OBJ 1-based).
///
/// @param filename     Path to the .obj file.
/// @param mesh         Output mesh (cleared first).
/// @param load_normals If true, load per-vertex normals and face normal indices.
/// @param load_uv      If true, load UV texture coordinates and face UV indices.
/// @returns true on success.
inline bool LoadOBJ(const std::string& filename, SurfaceMesh& mesh, bool load_normals = true, bool load_uv = false) {
    mesh.Clear();

    using namespace WAVEFRONT;
    GeometryInterface dummy;
    OBJ obj;
    if (obj.LoadMesh(filename.c_str(), &dummy, /*textured=*/true) == -1) {
        std::cerr << "LoadOBJ: failed to load: " << filename << "\n";
        return false;
    }

    // Vertices
    for (size_t i = 0; i + 2 < obj.mVerts.size(); i += 3)
        mesh.vertices.push_back(Real3d(obj.mVerts[i], obj.mVerts[i + 1], obj.mVerts[i + 2]));

    // Normals
    if (load_normals && !obj.mNormals.empty()) {
        for (size_t i = 0; i + 2 < obj.mNormals.size(); i += 3)
            mesh.normals.push_back(Real3d(obj.mNormals[i], obj.mNormals[i + 1], obj.mNormals[i + 2]));
    }

    // UVs
    if (load_uv && !obj.mTexels.empty()) {
        for (size_t i = 0; i + 1 < obj.mTexels.size(); i += 2)
            mesh.uvs.push_back(Real3d(obj.mTexels[i], obj.mTexels[i + 1], 0.0));
    }

    // Face vertex indices
    for (size_t i = 0; i + 2 < obj.mIndexesVerts.size(); i += 3)
        mesh.faces.push_back({obj.mIndexesVerts[i], obj.mIndexesVerts[i + 1], obj.mIndexesVerts[i + 2]});

    // Face normal indices (only if sizes are consistent)
    if (load_normals && obj.mIndexesNormals.size() == obj.mIndexesVerts.size()) {
        for (size_t i = 0; i + 2 < obj.mIndexesNormals.size(); i += 3)
            mesh.faceNormalIndices.push_back(
                {obj.mIndexesNormals[i], obj.mIndexesNormals[i + 1], obj.mIndexesNormals[i + 2]});
    }

    // Face UV indices (only if sizes are consistent)
    if (load_uv && obj.mIndexesTexels.size() == obj.mIndexesVerts.size()) {
        for (size_t i = 0; i + 2 < obj.mIndexesTexels.size(); i += 3)
            mesh.faceUVIndices.push_back({obj.mIndexesTexels[i], obj.mIndexesTexels[i + 1], obj.mIndexesTexels[i + 2]});
    }

    if (mesh.faces.empty()) {
        std::cerr << "LoadOBJ: no faces loaded from: " << filename << "\n";
        return false;
    }
    return true;
}

/// Write one or more surface meshes to a Wavefront OBJ file.
///
/// All meshes are written into a single OBJ file with global vertex/normal
/// offset bookkeeping.  If a mesh has normals, faces are written in v//vn
/// format; otherwise plain v format is used.
///
/// @param filename  Path to the output .obj file.
/// @param meshes    Meshes to write.
inline void WriteOBJ(const std::string& filename, const std::vector<SurfaceMesh>& meshes) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "WriteOBJ: cannot open file for writing: " << filename << "\n";
        return;
    }
    file << std::scientific;
    file.precision(10);

    int v_off = 1, vn_off = 1;
    for (size_t mi = 0; mi < meshes.size(); ++mi) {
        const auto& m = meshes[mi];
        file << "# mesh " << mi << "\n";

        for (const auto& v : m.vertices)
            file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";

        for (const auto& n : m.normals)
            file << "vn " << n.x() << " " << n.y() << " " << n.z() << "\n";

        if (m.HasNormals()) {
            assert(m.faces.size() == m.faceNormalIndices.size());
            for (size_t fi = 0; fi < m.faces.size(); ++fi) {
                const auto& fv = m.faces[fi];
                const auto& fn = m.faceNormalIndices[fi];
                file << "f " << (fv[0] + v_off) << "//" << (fn[0] + vn_off) << " " << (fv[1] + v_off) << "//"
                     << (fn[1] + vn_off) << " " << (fv[2] + v_off) << "//" << (fn[2] + vn_off) << "\n";
            }
        } else {
            for (const auto& fv : m.faces)
                file << "f " << (fv[0] + v_off) << " " << (fv[1] + v_off) << " " << (fv[2] + v_off) << "\n";
        }

        v_off += static_cast<int>(m.vertices.size());
        vn_off += static_cast<int>(m.normals.size());
    }
}

/// Write a single surface mesh to a Wavefront OBJ file.
inline void WriteOBJ(const std::string& filename, const SurfaceMesh& mesh) {
    WriteOBJ(filename, std::vector<SurfaceMesh>{mesh});
}

}  // namespace mophi

#endif
