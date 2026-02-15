//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cassert>
#include <utils/MeshIO.hpp>
#include <common/Mesh.hpp>

int main() {
    std::cout << "=== Testing Mesh I/O with beam.vtu ===" << std::endl;

    // Path to the beam mesh file
    std::string beamMeshPath = "../data/beam.vtu";
    std::string outputMeshPath = "/tmp/beam_output.vtu";

    // -----------------------------------------------------------------------
    // 1. Load the mesh
    // -----------------------------------------------------------------------
    std::cout << "\n[1] Loading mesh from: " << beamMeshPath << std::endl;
    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(beamMeshPath);
        std::cout << "    Successfully loaded mesh!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    ERROR: Failed to load mesh: " << e.what() << std::endl;
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
    std::cout << "    Number of owned hexes: " << mesh.NumOwnedHexes() << std::endl;
    std::cout << "    Part ID: " << mesh.partID << std::endl;

    // Verify we have some data
    assert(mesh.NumLocalNodes() > 0 && "Mesh should have nodes");
    assert(mesh.NumOwnedCells() > 0 && "Mesh should have cells");
    std::cout << "    ✓ Basic mesh structure is valid" << std::endl;

    // Verify node coordinates are loaded
    if (mesh.geom.nodes.size() > 0) {
        std::cout << "    First node position: (" 
                  << mesh.geom.nodes[0].x() << ", "
                  << mesh.geom.nodes[0].y() << ", "
                  << mesh.geom.nodes[0].z() << ")" << std::endl;
    }

    // Verify connectivity data
    if (mesh.topo.tets.size() > 0) {
        std::cout << "    First tet connectivity: ["
                  << mesh.topo.tets[0][0] << ", "
                  << mesh.topo.tets[0][1] << ", "
                  << mesh.topo.tets[0][2] << ", "
                  << mesh.topo.tets[0][3] << "]" << std::endl;
        
        // Verify all node indices are valid
        for (const auto& tet : mesh.topo.tets) {
            for (int i = 0; i < 4; ++i) {
                assert(tet[i] >= 0 && tet[i] < (mophi::nodeID_t)mesh.geom.nodes.size() 
                       && "Tet node index out of bounds");
            }
        }
        std::cout << "    ✓ All tet connectivity indices are valid" << std::endl;
    }

    if (mesh.topo.hexes.size() > 0) {
        std::cout << "    First hex connectivity: ["
                  << mesh.topo.hexes[0][0] << ", "
                  << mesh.topo.hexes[0][1] << ", "
                  << mesh.topo.hexes[0][2] << ", "
                  << mesh.topo.hexes[0][3] << ", "
                  << mesh.topo.hexes[0][4] << ", "
                  << mesh.topo.hexes[0][5] << ", "
                  << mesh.topo.hexes[0][6] << ", "
                  << mesh.topo.hexes[0][7] << "]" << std::endl;
        
        // Verify all node indices are valid
        for (const auto& hex : mesh.topo.hexes) {
            for (int i = 0; i < 8; ++i) {
                assert(hex[i] >= 0 && hex[i] < (mophi::nodeID_t)mesh.geom.nodes.size() 
                       && "Hex node index out of bounds");
            }
        }
        std::cout << "    ✓ All hex connectivity indices are valid" << std::endl;
    }

    // Verify halo data structure
    std::cout << "    Global to local node map size: " 
              << mesh.halo.globalToLocalNode.size() << std::endl;
    std::cout << "    Local to global node vector size: " 
              << mesh.halo.localToGlobalNode.size() << std::endl;
    assert(mesh.halo.localToGlobalNode.size() == mesh.geom.nodes.size() 
           && "Local to global node mapping size mismatch");
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
    assert(reloadedMesh.NumLocalNodes() == mesh.NumLocalNodes() 
           && "Reloaded mesh node count mismatch");
    assert(reloadedMesh.NumOwnedNodes() == mesh.NumOwnedNodes() 
           && "Reloaded mesh owned node count mismatch");
    assert(reloadedMesh.NumOwnedTets() == mesh.NumOwnedTets() 
           && "Reloaded mesh tet count mismatch");
    assert(reloadedMesh.NumOwnedHexes() == mesh.NumOwnedHexes() 
           && "Reloaded mesh hex count mismatch");
    assert(reloadedMesh.partID == mesh.partID 
           && "Reloaded mesh part ID mismatch");

    std::cout << "    ✓ Reloaded mesh structure matches original" << std::endl;

    // Verify node coordinates match
    for (size_t i = 0; i < mesh.geom.nodes.size(); ++i) {
        double dx = mesh.geom.nodes[i].x() - reloadedMesh.geom.nodes[i].x();
        double dy = mesh.geom.nodes[i].y() - reloadedMesh.geom.nodes[i].y();
        double dz = mesh.geom.nodes[i].z() - reloadedMesh.geom.nodes[i].z();
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        assert(dist < 1e-10 && "Node coordinate mismatch after reload");
    }
    std::cout << "    ✓ Node coordinates match" << std::endl;

    // Verify connectivity matches
    for (size_t i = 0; i < mesh.topo.tets.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            assert(mesh.topo.tets[i][j] == reloadedMesh.topo.tets[i][j] 
                   && "Tet connectivity mismatch after reload");
        }
    }
    for (size_t i = 0; i < mesh.topo.hexes.size(); ++i) {
        for (int j = 0; j < 8; ++j) {
            assert(mesh.topo.hexes[i][j] == reloadedMesh.topo.hexes[i][j] 
                   && "Hex connectivity mismatch after reload");
        }
    }
    std::cout << "    ✓ Connectivity data matches" << std::endl;

    // -----------------------------------------------------------------------
    // 5. Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Successfully loaded beam.vtu" << std::endl;
    std::cout << "  - Verified mesh data structure integrity" << std::endl;
    std::cout << "  - Successfully wrote mesh to " << outputMeshPath << std::endl;
    std::cout << "  - Verified write/read consistency" << std::endl;

    return 0;
}
