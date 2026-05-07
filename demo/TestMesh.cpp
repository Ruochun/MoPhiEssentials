//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <filesystem>
#include "utils/MeshIO.hpp"
#include "common/Mesh.hpp"
#include "core/Quaternion.hpp"

int main() {
    std::cout << "=== Testing Mesh Functionality ===" << std::endl;

    // Try multiple mesh files - beam.vtu is binary (may not be supported yet),
    // cube_mesh.vtu is ASCII and should work, and beam_tet10.vtu tests tet10 support
    std::vector<std::string> meshFiles = {"../data/beam_tet10.vtu", "../data/beam.vtu"};

    std::string loadedMeshPath;
    const std::string outputDir = "./test_output/mesh";
    std::filesystem::create_directories(outputDir);
    std::string outputMeshPath = outputDir + "/mesh_output.vtu";
    mophi::Mesh mesh;

    // -----------------------------------------------------------------------
    // 1. Load the mesh - try each file until one succeeds
    // -----------------------------------------------------------------------
    std::cout << "\n[1] Loading mesh..." << std::endl;
    bool loaded = false;
    for (const auto& meshPath : meshFiles) {
        std::cout << "    Trying: " << meshPath << std::endl;
        try {
            mesh = mophi::LoadVtu(meshPath);
            loadedMeshPath = meshPath;
            loaded = true;
            std::cout << "    Successfully loaded mesh from: " << meshPath << std::endl;
            break;
        } catch (const std::exception& e) {
            std::cerr << "    WARNING: Could not load " << meshPath << ": " << e.what() << std::endl;
            std::cerr << "    (This may be expected if the file uses unsupported binary/compressed format)"
                      << std::endl;
        }
    }

    if (!loaded) {
        std::cerr << "\nERROR: Could not load any test mesh files!" << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. Verify the loaded mesh data structure
    // -----------------------------------------------------------------------
    std::cout << "\n[2] Verifying loaded mesh data structure:" << std::endl;

    // Check basic counts
    std::cout << "    Number of owned nodes: " << mesh.NumOwnedNodes() << std::endl;
    std::cout << "    Number of local nodes: " << mesh.NumLocalNodes() << std::endl;
    std::cout << "    Number of owned cells: " << mesh.NumOwnedCells() << std::endl;
    std::cout << "    Number of owned tets:  " << mesh.NumOwnedTets() << std::endl;
    std::cout << "    Number of owned tet10s: " << mesh.NumOwnedTet10s() << std::endl;
    std::cout << "    Number of owned hexes: " << mesh.NumOwnedHexes() << std::endl;
    std::cout << "    Part ID: " << mesh.partID << std::endl;

    // Verify we have some data
    assert(mesh.NumLocalNodes() > 0 && "Mesh should have nodes");
    assert(mesh.NumOwnedCells() > 0 && "Mesh should have cells");
    std::cout << "    ✓ Basic mesh structure is valid" << std::endl;

    // Verify node coordinates are loaded
    if (mesh.geom.nodes.size() > 0) {
        std::cout << "    First node position: (" << mesh.geom.nodes[0].x() << ", " << mesh.geom.nodes[0].y() << ", "
                  << mesh.geom.nodes[0].z() << ")" << std::endl;
    }

    // Verify connectivity data
    if (mesh.topo.tets.size() > 0) {
        std::cout << "    First tet connectivity: [" << mesh.topo.tets[0][0] << ", " << mesh.topo.tets[0][1] << ", "
                  << mesh.topo.tets[0][2] << ", " << mesh.topo.tets[0][3] << "]" << std::endl;

        // Verify all node indices are valid (nodeID_t is signed, so check both bounds)
        for (const auto& tet : mesh.topo.tets) {
            for (int i = 0; i < 4; ++i) {
                assert(tet[i] >= 0 && tet[i] < static_cast<mophi::nodeID_t>(mesh.geom.nodes.size()) &&
                       "Tet node index out of bounds");
            }
        }
        std::cout << "    ✓ All tet connectivity indices are valid" << std::endl;
    }

    if (mesh.topo.tet10s.size() > 0) {
        std::cout << "    First tet10 connectivity: [" << mesh.topo.tet10s[0][0] << ", " << mesh.topo.tet10s[0][1]
                  << ", " << mesh.topo.tet10s[0][2] << ", " << mesh.topo.tet10s[0][3] << ", " << mesh.topo.tet10s[0][4]
                  << ", " << mesh.topo.tet10s[0][5] << ", " << mesh.topo.tet10s[0][6] << ", " << mesh.topo.tet10s[0][7]
                  << ", " << mesh.topo.tet10s[0][8] << ", " << mesh.topo.tet10s[0][9] << "]" << std::endl;

        // Verify all node indices are valid (nodeID_t is signed, so check both bounds)
        for (const auto& tet10 : mesh.topo.tet10s) {
            for (int i = 0; i < 10; ++i) {
                assert(tet10[i] >= 0 && tet10[i] < static_cast<mophi::nodeID_t>(mesh.geom.nodes.size()) &&
                       "Tet10 node index out of bounds");
            }
        }
        std::cout << "    ✓ All tet10 connectivity indices are valid" << std::endl;
    }

    if (mesh.topo.hexes.size() > 0) {
        std::cout << "    First hex connectivity: [" << mesh.topo.hexes[0][0] << ", " << mesh.topo.hexes[0][1] << ", "
                  << mesh.topo.hexes[0][2] << ", " << mesh.topo.hexes[0][3] << ", " << mesh.topo.hexes[0][4] << ", "
                  << mesh.topo.hexes[0][5] << ", " << mesh.topo.hexes[0][6] << ", " << mesh.topo.hexes[0][7] << "]"
                  << std::endl;

        // Verify all node indices are valid (nodeID_t is signed, so check both bounds)
        for (const auto& hex : mesh.topo.hexes) {
            for (int i = 0; i < 8; ++i) {
                assert(hex[i] >= 0 && hex[i] < static_cast<mophi::nodeID_t>(mesh.geom.nodes.size()) &&
                       "Hex node index out of bounds");
            }
        }
        std::cout << "    ✓ All hex connectivity indices are valid" << std::endl;
    }

    // Verify halo data structure
    std::cout << "    Global to local node map size: " << mesh.halo.globalToLocalNode.size() << std::endl;
    std::cout << "    Local to global node vector size: " << mesh.halo.localToGlobalNode.size() << std::endl;
    assert(mesh.halo.localToGlobalNode.size() == mesh.geom.nodes.size() &&
           "Local to global node mapping size mismatch");
    std::cout << "    ✓ Halo data structure is consistent" << std::endl;

    // -----------------------------------------------------------------------
    // 3. Write the mesh to a new file
    // -----------------------------------------------------------------------
    std::cout << "\n[3] Writing mesh to: " << outputMeshPath << std::endl;
    try {
        mophi::WriteVtu(outputMeshPath, mesh);
        std::cout << "    Successfully wrote mesh!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    ERROR: Failed to write mesh: " << e.what() << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 4. Reload the written mesh to verify write/read consistency
    // -----------------------------------------------------------------------
    std::cout << "\n[4] Reloading written mesh to verify consistency:" << std::endl;
    mophi::Mesh reloadedMesh;
    try {
        reloadedMesh = mophi::LoadVtu(outputMeshPath);
        std::cout << "    Successfully reloaded mesh!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    ERROR: Failed to reload mesh: " << e.what() << std::endl;
        return 1;
    }

    // Verify the reloaded mesh matches the original
    assert(reloadedMesh.NumLocalNodes() == mesh.NumLocalNodes() && "Reloaded mesh node count mismatch");
    assert(reloadedMesh.NumOwnedNodes() == mesh.NumOwnedNodes() && "Reloaded mesh owned node count mismatch");
    assert(reloadedMesh.NumOwnedTets() == mesh.NumOwnedTets() && "Reloaded mesh tet count mismatch");
    assert(reloadedMesh.NumOwnedTet10s() == mesh.NumOwnedTet10s() && "Reloaded mesh tet10 count mismatch");
    assert(reloadedMesh.NumOwnedHexes() == mesh.NumOwnedHexes() && "Reloaded mesh hex count mismatch");
    assert(reloadedMesh.partID == mesh.partID && "Reloaded mesh part ID mismatch");

    std::cout << "    ✓ Reloaded mesh structure matches original" << std::endl;

    // Verify node coordinates match
    for (size_t i = 0; i < mesh.geom.nodes.size(); ++i) {
        double dx = mesh.geom.nodes[i].x() - reloadedMesh.geom.nodes[i].x();
        double dy = mesh.geom.nodes[i].y() - reloadedMesh.geom.nodes[i].y();
        double dz = mesh.geom.nodes[i].z() - reloadedMesh.geom.nodes[i].z();
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        assert(dist < 1e-10 && "Node coordinate mismatch after reload");
    }
    std::cout << "    ✓ Node coordinates match" << std::endl;

    // Verify connectivity matches
    for (size_t i = 0; i < mesh.topo.tets.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            assert(mesh.topo.tets[i][j] == reloadedMesh.topo.tets[i][j] && "Tet connectivity mismatch after reload");
        }
    }
    for (size_t i = 0; i < mesh.topo.tet10s.size(); ++i) {
        for (int j = 0; j < 10; ++j) {
            assert(mesh.topo.tet10s[i][j] == reloadedMesh.topo.tet10s[i][j] &&
                   "Tet10 connectivity mismatch after reload");
        }
    }
    for (size_t i = 0; i < mesh.topo.hexes.size(); ++i) {
        for (int j = 0; j < 8; ++j) {
            assert(mesh.topo.hexes[i][j] == reloadedMesh.topo.hexes[i][j] && "Hex connectivity mismatch after reload");
        }
    }
    std::cout << "    ✓ Connectivity data matches" << std::endl;

    // -----------------------------------------------------------------------
    // 5. Test mesh geometric utilities
    // -----------------------------------------------------------------------
    std::cout << "\n[5] Testing mesh geometric utilities:" << std::endl;
    const double pi = std::acos(-1.0);
    const double tol = 1e-10;

    // Helper lambda to compare doubles with tolerance
    auto near = [&](double a, double b) { return std::abs(a - b) < tol; };
    auto near3 = [&](const mophi::Real3d& a, const mophi::Real3d& b) {
        return near(a.x(), b.x()) && near(a.y(), b.y()) && near(a.z(), b.z());
    };

    // 5a. Translate
    {
        mophi::Mesh m = mesh;
        mophi::Real3d offset(1.0, 2.0, 3.0);
        m.Translate(offset);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            mophi::Real3d expected = mesh.geom.nodes[i] + offset;
            assert(near3(m.geom.nodes[i], expected) && "Translate: node mismatch");
        }
        std::cout << "    ✓ Translate moves all nodes by the given offset" << std::endl;
    }

    // 5b. Scale (uniform)
    {
        mophi::Mesh m = mesh;
        double factor = 2.5;
        m.Scale(factor);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            mophi::Real3d expected = mesh.geom.nodes[i] * factor;
            assert(near3(m.geom.nodes[i], expected) && "Scale (uniform): node mismatch");
        }
        std::cout << "    ✓ Scale (uniform) scales all nodes by the given factor" << std::endl;
    }

    // 5c. Scale (per-axis)
    {
        mophi::Mesh m = mesh;
        double sx = 2.0, sy = 3.0, sz = 0.5;
        m.Scale(sx, sy, sz);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            const mophi::Real3d& orig = mesh.geom.nodes[i];
            mophi::Real3d expected(orig.x() * sx, orig.y() * sy, orig.z() * sz);
            assert(near3(m.geom.nodes[i], expected) && "Scale (per-axis): node mismatch");
        }
        std::cout << "    ✓ Scale (per-axis) scales each axis independently" << std::endl;
    }

    // 5d. Rotate (90° around Z axis)
    {
        mophi::Mesh m = mesh;
        mophi::Real3d zAxis(0, 0, 1);
        mophi::Quatd q = mophi::Quatd::FromAxisAngle(zAxis, pi / 2.0);
        m.Rotate(q);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            mophi::Real3d expected = q.Rotate(mesh.geom.nodes[i]);
            assert(near3(m.geom.nodes[i], expected) && "Rotate: node mismatch");
        }
        std::cout << "    ✓ Rotate applies quaternion rotation to all nodes" << std::endl;
    }

    // 5e. RotateAndTranslate
    {
        mophi::Mesh m = mesh;
        mophi::Real3d yAxis(0, 1, 0);
        mophi::Quatd q = mophi::Quatd::FromAxisAngle(yAxis, pi / 4.0);
        mophi::Real3d offset(-1.0, 0.5, 2.0);
        m.RotateAndTranslate(q, offset);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            mophi::Real3d expected = q.Rotate(mesh.geom.nodes[i]) + offset;
            assert(near3(m.geom.nodes[i], expected) && "RotateAndTranslate: node mismatch");
        }
        std::cout << "    ✓ RotateAndTranslate applies rotation then translation" << std::endl;
    }

    // 5f. Mirror across the XZ-plane (normal = (0,1,0), point = origin)
    // Mirroring flips y: node (x,y,z) -> (x,-y,z)
    {
        mophi::Mesh m = mesh;
        mophi::Real3d normal(0, 1, 0);
        m.Mirror(normal);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            const mophi::Real3d& orig = mesh.geom.nodes[i];
            mophi::Real3d expected(orig.x(), -orig.y(), orig.z());
            assert(near3(m.geom.nodes[i], expected) && "Mirror (XZ-plane): node mismatch");
        }
        std::cout << "    ✓ Mirror across XZ-plane flips y-coordinates" << std::endl;
    }

    // 5g. Mirror with non-origin reference point
    // Mirror across plane x=1 (normal=(1,0,0), point=(1,0,0)):
    // node (x,y,z) -> (2 - x, y, z)
    {
        mophi::Mesh m = mesh;
        mophi::Real3d normal(1, 0, 0);
        mophi::Real3d point(1, 0, 0);
        m.Mirror(normal, point);
        for (size_t i = 0; i < m.geom.nodes.size(); ++i) {
            const mophi::Real3d& orig = mesh.geom.nodes[i];
            mophi::Real3d expected(2.0 - orig.x(), orig.y(), orig.z());
            assert(near3(m.geom.nodes[i], expected) && "Mirror (x=1 plane): node mismatch");
        }
        std::cout << "    ✓ Mirror across x=1 plane reflects x-coordinates correctly" << std::endl;
    }

    // 5h. Connectivity is unchanged after all transformations
    {
        mophi::Mesh m = mesh;
        mophi::Real3d zAxis(0, 0, 1);
        mophi::Quatd q = mophi::Quatd::FromAxisAngle(zAxis, pi / 6.0);
        m.Rotate(q);
        m.Translate(mophi::Real3d(1, 2, 3));
        m.Scale(2.0);

        // Connectivity must be identical to the original
        assert(m.topo.tets.size() == mesh.topo.tets.size() && "Connectivity size mismatch after transforms");
        assert(m.topo.tet10s.size() == mesh.topo.tet10s.size() && "Connectivity size mismatch after transforms");
        assert(m.topo.hexes.size() == mesh.topo.hexes.size() && "Connectivity size mismatch after transforms");
        for (size_t i = 0; i < m.topo.tet10s.size(); ++i) {
            for (int j = 0; j < 10; ++j)
                assert(m.topo.tet10s[i][j] == mesh.topo.tet10s[i][j] && "Connectivity changed after transforms");
        }
        std::cout << "    ✓ Connectivity is unchanged after geometric transforms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 6. Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Successfully loaded " << loadedMeshPath << std::endl;
    std::cout << "  - Verified mesh data structure integrity" << std::endl;
    std::cout << "  - Successfully wrote mesh to " << outputMeshPath << std::endl;
    std::cout << "  - Verified write/read consistency" << std::endl;
    std::cout << "  - Verified Translate, Scale, Rotate, RotateAndTranslate, Mirror utilities" << std::endl;

    return 0;
}
