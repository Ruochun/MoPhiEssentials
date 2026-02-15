//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <utils/MeshIO.hpp>
#include <common/Mesh.hpp>

int main() {
    std::cout << "=== Testing Mesh I/O Functionality ===" << std::endl;

    // Try multiple mesh files - beam.vtu is binary (may not be supported yet),
    // cube_mesh.vtu is ASCII and should work, and beam_tet10.vtu tests tet10 support
    std::vector<std::string> meshFiles = {"../data/beam_tet10.vtu", "../data/beam.vtu"};

    std::string loadedMeshPath;
    std::string outputMeshPath = "mesh_output.vtu";  // Use relative path in current directory
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
    // 5. Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Successfully loaded " << loadedMeshPath << std::endl;
    std::cout << "  - Verified mesh data structure integrity" << std::endl;
    std::cout << "  - Successfully wrote mesh to " << outputMeshPath << std::endl;
    std::cout << "  - Verified write/read consistency" << std::endl;

    return 0;
}
