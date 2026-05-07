//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

/// Demo / unit-test for the SurfaceMesh I/O utilities (STL, PLY, OBJ) and the
/// mesh analysis utilities (IsWatertight, ComputeMassProperties,
/// BuildAdjacencyWithEdgeInfo, ComputeFaceNormals).

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "common/Mesh.hpp"
#include "utils/MeshIO.hpp"

// ---------------------------------------------------------------------------
// Build a unit right-angle tetrahedron with outward-facing winding.
//   v0 = (0,0,0)   v1 = (1,0,0)   v2 = (0,1,0)   v3 = (0,0,1)
//   Volume  = 1/6
//   CoM     = (1/4, 1/4, 1/4)
//   Ixx_CoM = Iyy_CoM = Izz_CoM = 1/80  (unit density)
// ---------------------------------------------------------------------------
static mophi::SurfaceMesh make_unit_tet() {
    mophi::SurfaceMesh m;
    m.vertices = {
        mophi::Real3d(0, 0, 0),
        mophi::Real3d(1, 0, 0),
        mophi::Real3d(0, 1, 0),
        mophi::Real3d(0, 0, 1),
    };
    // Outward-facing triangles (right-hand rule, normal pointing away from interior)
    m.faces = {
        {0, 2, 1},  // bottom face  (normal ~ -Z)
        {0, 1, 3},  // front face   (normal ~ -Y)
        {0, 3, 2},  // left face    (normal ~ -X)
        {1, 2, 3},  // slant face   (normal ~ (1,1,1)/sqrt(3))
    };
    return m;
}

// ---------------------------------------------------------------------------
// Build a unit cube (8 vertices, 12 triangles, side length = 1) whose
// centroid is at the origin.
// ---------------------------------------------------------------------------
static mophi::SurfaceMesh make_unit_cube() {
    mophi::SurfaceMesh m;
    // 8 corners of [-0.5, 0.5]^3
    m.vertices = {
        mophi::Real3d(-0.5, -0.5, -0.5),  // 0
        mophi::Real3d(+0.5, -0.5, -0.5),  // 1
        mophi::Real3d(+0.5, +0.5, -0.5),  // 2
        mophi::Real3d(-0.5, +0.5, -0.5),  // 3
        mophi::Real3d(-0.5, -0.5, +0.5),  // 4
        mophi::Real3d(+0.5, -0.5, +0.5),  // 5
        mophi::Real3d(+0.5, +0.5, +0.5),  // 6
        mophi::Real3d(-0.5, +0.5, +0.5),  // 7
    };
    // Each face is split into 2 triangles, winding chosen so normals point out.
    m.faces = {
        // -Z face
        {0, 3, 2},
        {0, 2, 1},
        // +Z face
        {4, 5, 6},
        {4, 6, 7},
        // -Y face
        {0, 1, 5},
        {0, 5, 4},
        // +Y face
        {3, 7, 6},
        {3, 6, 2},
        // -X face
        {0, 4, 7},
        {0, 7, 3},
        // +X face
        {1, 2, 6},
        {1, 6, 5},
    };
    return m;
}

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------
static bool near(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) < tol;
}
static bool near3(const mophi::Real3d& a, const mophi::Real3d& b, double tol = 1e-9) {
    return near(a.x(), b.x(), tol) && near(a.y(), b.y(), tol) && near(a.z(), b.z(), tol);
}

int main() {
    std::cout << "=== TestSurfaceMeshIO ===" << std::endl;

    // -----------------------------------------------------------------------
    // 1. Basic SurfaceMesh construction + ComputeFaceNormals
    // -----------------------------------------------------------------------
    std::cout << "\n[1] SurfaceMesh construction and ComputeFaceNormals" << std::endl;
    {
        mophi::SurfaceMesh tet = make_unit_tet();
        assert(tet.NumVertices() == 4);
        assert(tet.NumFaces() == 4);
        assert(!tet.HasNormals());

        tet.ComputeFaceNormals();
        assert(tet.HasNormals());
        assert(tet.normals.size() == 4);
        // Bottom face {0,2,1}: n = (v2-v0) x (v1-v0) = (0,1,0) x (1,0,0) = (0,0,-1)
        assert(near3(tet.normals[0], mophi::Real3d(0, 0, -1)));
        std::cout << "  ✓ ComputeFaceNormals produces correct normals" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Geometric transforms on SurfaceMesh
    // -----------------------------------------------------------------------
    std::cout << "\n[2] Geometric transforms on SurfaceMesh" << std::endl;
    {
        mophi::SurfaceMesh tet = make_unit_tet();

        // Translate
        mophi::SurfaceMesh t = tet;
        t.Translate(mophi::Real3d(1, 2, 3));
        assert(near3(t.vertices[0], mophi::Real3d(1, 2, 3)));
        assert(near3(t.vertices[1], mophi::Real3d(2, 2, 3)));
        std::cout << "  ✓ Translate" << std::endl;

        // Scale uniform
        t = tet;
        t.Scale(2.0);
        assert(near3(t.vertices[1], mophi::Real3d(2, 0, 0)));
        std::cout << "  ✓ Scale (uniform)" << std::endl;

        // Scale per-axis
        t = tet;
        t.Scale(1.0, 2.0, 3.0);
        assert(near3(t.vertices[2], mophi::Real3d(0, 2, 0)));
        assert(near3(t.vertices[3], mophi::Real3d(0, 0, 3)));
        std::cout << "  ✓ Scale (per-axis)" << std::endl;

        // Mirror across XY plane (normal = (0,0,1))
        t = tet;
        t.Mirror(mophi::Real3d(0, 0, 1));
        assert(near3(t.vertices[3], mophi::Real3d(0, 0, -1)));
        std::cout << "  ✓ Mirror" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. IsWatertight
    // -----------------------------------------------------------------------
    std::cout << "\n[3] IsWatertight" << std::endl;
    {
        mophi::SurfaceMesh tet = make_unit_tet();
        size_t be = 0, nme = 0;
        bool wt = tet.IsWatertight(&be, &nme);
        std::cout << "  Tet: watertight=" << wt << "  boundary=" << be
                  << "  non-manifold=" << nme << std::endl;
        assert(wt && "Unit tet should be watertight");
        assert(be == 0 && nme == 0);
        std::cout << "  ✓ Closed tetrahedron is watertight" << std::endl;

        mophi::SurfaceMesh cube = make_unit_cube();
        be = nme = 0;
        wt = cube.IsWatertight(&be, &nme);
        std::cout << "  Cube: watertight=" << wt << "  boundary=" << be
                  << "  non-manifold=" << nme << std::endl;
        assert(wt && "Unit cube should be watertight");
        std::cout << "  ✓ Closed cube is watertight" << std::endl;

        // Open mesh: remove one face from the tet → no longer watertight
        mophi::SurfaceMesh open_tet = tet;
        open_tet.faces.pop_back();
        be = nme = 0;
        wt = open_tet.IsWatertight(&be, &nme);
        assert(!wt && "Open tet should NOT be watertight");
        assert(be > 0);
        std::cout << "  ✓ Open mesh is correctly identified as not watertight" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4. ComputeMassProperties (unit tetrahedron, known analytical values)
    // -----------------------------------------------------------------------
    std::cout << "\n[4] ComputeMassProperties" << std::endl;
    {
        mophi::SurfaceMesh tet = make_unit_tet();
        double volume;
        mophi::Real3d center, inertia, inertia_products;
        tet.ComputeMassProperties(volume, center, inertia, inertia_products);

        std::cout << "  Volume     = " << volume << "  (expected " << 1.0 / 6.0 << ")\n";
        std::cout << "  CoM        = (" << center.x() << ", " << center.y() << ", " << center.z()
                  << ")  (expected (0.25, 0.25, 0.25))\n";
        std::cout << "  Inertia    = (" << inertia.x() << ", " << inertia.y() << ", "
                  << inertia.z() << ")  (expected ~0.0125 each)\n";

        assert(near(volume, 1.0 / 6.0, 1e-10) && "Volume mismatch for unit tet");
        assert(near3(center, mophi::Real3d(0.25, 0.25, 0.25), 1e-10) &&
               "CoM mismatch for unit tet");
        // Ixx_CoM = 1/80 = 0.0125
        assert(near(inertia.x(), 1.0 / 80.0, 1e-10) && "Ixx mismatch for unit tet");
        assert(near(inertia.y(), 1.0 / 80.0, 1e-10) && "Iyy mismatch for unit tet");
        assert(near(inertia.z(), 1.0 / 80.0, 1e-10) && "Izz mismatch for unit tet");
        std::cout << "  ✓ Mass properties match analytical values for unit tetrahedron\n";

        // Unit cube centred at origin: volume = 1, CoM = (0,0,0), Ixx=Iyy=Izz = 1/6
        mophi::SurfaceMesh cube = make_unit_cube();
        double vol2;
        mophi::Real3d ctr2, iner2, prod2;
        cube.ComputeMassProperties(vol2, ctr2, iner2, prod2);
        std::cout << "  Cube volume = " << vol2 << "  (expected 1.0)\n";
        std::cout << "  Cube CoM    = (" << ctr2.x() << ", " << ctr2.y() << ", " << ctr2.z()
                  << ")  (expected (0,0,0))\n";
        std::cout << "  Cube Inertia= (" << iner2.x() << ", " << iner2.y() << ", " << iner2.z()
                  << ")  (expected ~0.1667 each)\n";
        assert(near(vol2, 1.0, 1e-10) && "Volume mismatch for unit cube");
        assert(near3(ctr2, mophi::Real3d(0, 0, 0), 1e-10) && "CoM mismatch for unit cube");
        assert(near(iner2.x(), 1.0 / 6.0, 1e-10) && "Ixx mismatch for unit cube");
        assert(near(iner2.y(), 1.0 / 6.0, 1e-10) && "Iyy mismatch for unit cube");
        assert(near(iner2.z(), 1.0 / 6.0, 1e-10) && "Izz mismatch for unit cube");
        std::cout << "  ✓ Mass properties match analytical values for unit cube\n";
    }

    // -----------------------------------------------------------------------
    // 5. BuildAdjacencyWithEdgeInfo
    // -----------------------------------------------------------------------
    std::cout << "\n[5] BuildAdjacencyWithEdgeInfo" << std::endl;
    {
        mophi::SurfaceMesh tet = make_unit_tet();
        auto adj = tet.BuildAdjacencyWithEdgeInfo();
        assert(adj.size() == 4);
        // Each face of a closed tet has exactly 3 neighbours
        for (size_t i = 0; i < 4; ++i) {
            assert(adj[i].size() == 3 && "Each tet face should have 3 neighbours");
        }
        // For a consistently-wound mesh every shared edge should be oriented_ok
        for (size_t i = 0; i < 4; ++i) {
            for (const auto& info : adj[i]) {
                assert(info.oriented_ok && "Tet mesh should be consistently oriented");
            }
        }
        std::cout << "  ✓ Each tet face has exactly 3 consistently-oriented neighbours\n";
    }

    // -----------------------------------------------------------------------
    // 6. STL round-trip (binary)
    // -----------------------------------------------------------------------
    std::cout << "\n[6] STL binary round-trip" << std::endl;
    {
        const std::string stl_file = "/tmp/test_surface_mesh.stl";
        mophi::SurfaceMesh orig = make_unit_tet();
        assert(WriteSTL(stl_file, orig, /*binary=*/true));

        mophi::SurfaceMesh reloaded;
        assert(LoadSTL(stl_file, reloaded, /*load_normals=*/true));

        assert(reloaded.NumFaces() == orig.NumFaces() && "STL face count mismatch");
        // STL duplicates vertices per triangle: each face = 3 unique vertices
        assert(reloaded.NumVertices() == orig.NumFaces() * 3 &&
               "STL vertex count mismatch");
        assert(reloaded.HasNormals() && "STL reload should have normals (load_normals=true)");

        // Verify vertex positions match (each face's 3 consecutive vertices)
        const double tol = 1e-5;  // float precision from STL
        for (size_t fi = 0; fi < orig.NumFaces(); ++fi) {
            const auto& of = orig.faces[fi];
            // reloaded face fi -> vertices 3*fi, 3*fi+1, 3*fi+2
            for (int k = 0; k < 3; ++k) {
                const mophi::Real3d& rv = reloaded.vertices[3 * fi + k];
                const mophi::Real3d& ov = orig.vertices[(size_t)of[k]];
                assert(near3(rv, ov, tol) && "STL vertex position mismatch");
            }
        }
        std::cout << "  ✓ STL binary write/read round-trip OK\n";
    }

    // -----------------------------------------------------------------------
    // 7. STL round-trip (ASCII)
    // -----------------------------------------------------------------------
    std::cout << "\n[7] STL ASCII round-trip" << std::endl;
    {
        const std::string stl_file = "/tmp/test_surface_mesh_ascii.stl";
        mophi::SurfaceMesh orig = make_unit_tet();
        assert(WriteSTL(stl_file, orig, /*binary=*/false));

        mophi::SurfaceMesh reloaded;
        assert(LoadSTL(stl_file, reloaded));
        assert(reloaded.NumFaces() == orig.NumFaces() && "ASCII STL face count mismatch");
        const double tol = 1e-7;
        for (size_t fi = 0; fi < orig.NumFaces(); ++fi) {
            const auto& of = orig.faces[fi];
            for (int k = 0; k < 3; ++k) {
                const mophi::Real3d& rv = reloaded.vertices[3 * fi + k];
                const mophi::Real3d& ov = orig.vertices[(size_t)of[k]];
                assert(near3(rv, ov, tol) && "ASCII STL vertex position mismatch");
            }
        }
        std::cout << "  ✓ STL ASCII write/read round-trip OK\n";
    }

    // -----------------------------------------------------------------------
    // 8. IsWatertight on STL-reloaded mesh (needs vertex welding)
    // -----------------------------------------------------------------------
    std::cout << "\n[8] IsWatertight on STL-reloaded mesh (quantised vertex welding)" << std::endl;
    {
        const std::string stl_file = "/tmp/test_surface_mesh_wt.stl";
        mophi::SurfaceMesh orig = make_unit_tet();
        assert(WriteSTL(stl_file, orig, /*binary=*/true));

        mophi::SurfaceMesh stl_mesh;
        assert(LoadSTL(stl_file, stl_mesh));
        size_t be = 0, nme = 0;
        bool wt = stl_mesh.IsWatertight(&be, &nme);
        std::cout << "  STL-reloaded tet: watertight=" << wt << "  boundary=" << be
                  << "  non-manifold=" << nme << "\n";
        assert(wt && "STL-reloaded tet should be watertight after vertex welding");
        std::cout << "  ✓ IsWatertight correctly welds duplicate STL vertices\n";
    }

    // -----------------------------------------------------------------------
    // 9. PLY ASCII round-trip
    // -----------------------------------------------------------------------
    std::cout << "\n[9] PLY ASCII round-trip" << std::endl;
    {
        const std::string ply_file = "/tmp/test_surface_mesh.ply";
        mophi::SurfaceMesh orig = make_unit_cube();
        assert(WritePLY(ply_file, orig, /*binary=*/false));

        mophi::SurfaceMesh reloaded;
        assert(LoadPLY(ply_file, reloaded, /*load_normals=*/false));
        assert(reloaded.NumVertices() == orig.NumVertices() && "PLY vertex count mismatch");
        assert(reloaded.NumFaces() == orig.NumFaces() && "PLY face count mismatch");

        const double tol = 1e-6;
        for (size_t i = 0; i < orig.NumVertices(); ++i)
            assert(near3(reloaded.vertices[i], orig.vertices[i], tol) &&
                   "PLY vertex position mismatch");
        for (size_t i = 0; i < orig.NumFaces(); ++i)
            for (int k = 0; k < 3; ++k)
                assert(reloaded.faces[i][k] == orig.faces[i][k] && "PLY face index mismatch");
        std::cout << "  ✓ PLY ASCII write/read round-trip OK\n";
    }

    // -----------------------------------------------------------------------
    // 10. PLY binary round-trip
    // -----------------------------------------------------------------------
    std::cout << "\n[10] PLY binary round-trip" << std::endl;
    {
        const std::string ply_file = "/tmp/test_surface_mesh_bin.ply";
        mophi::SurfaceMesh orig = make_unit_tet();
        assert(WritePLY(ply_file, orig, /*binary=*/true));

        mophi::SurfaceMesh reloaded;
        assert(LoadPLY(ply_file, reloaded));
        assert(reloaded.NumVertices() == orig.NumVertices() && "Binary PLY vertex count mismatch");
        assert(reloaded.NumFaces() == orig.NumFaces() && "Binary PLY face count mismatch");
        const double tol = 1e-5;  // float precision
        for (size_t i = 0; i < orig.NumVertices(); ++i)
            assert(near3(reloaded.vertices[i], orig.vertices[i], tol) &&
                   "Binary PLY vertex position mismatch");
        std::cout << "  ✓ PLY binary write/read round-trip OK\n";
    }

    // -----------------------------------------------------------------------
    // 11. PLY with load_normals=true
    // -----------------------------------------------------------------------
    std::cout << "\n[11] PLY load_normals=true (computes per-face normals)" << std::endl;
    {
        const std::string ply_file = "/tmp/test_surface_mesh_normals.ply";
        mophi::SurfaceMesh orig = make_unit_tet();
        assert(WritePLY(ply_file, orig, /*binary=*/false));

        mophi::SurfaceMesh reloaded;
        assert(LoadPLY(ply_file, reloaded, /*load_normals=*/true));
        assert(reloaded.HasNormals() && "Should have normals after load_normals=true");
        assert(reloaded.normals.size() == reloaded.NumFaces());
        // Bottom face normal should be ~(0,0,-1)
        assert(near3(reloaded.normals[0], mophi::Real3d(0, 0, -1)) &&
               "PLY normal[0] mismatch");
        std::cout << "  ✓ PLY load_normals=true computes correct face normals\n";
    }

    // -----------------------------------------------------------------------
    // 12. OBJ round-trip
    // -----------------------------------------------------------------------
    std::cout << "\n[12] OBJ round-trip" << std::endl;
    {
        const std::string obj_file = "/tmp/test_surface_mesh.obj";
        mophi::SurfaceMesh orig = make_unit_cube();
        orig.ComputeFaceNormals();  // add normals so they are written too
        WriteOBJ(obj_file, orig);

        mophi::SurfaceMesh reloaded;
        assert(LoadOBJ(obj_file, reloaded, /*load_normals=*/true));

        // OBJ preserves topology exactly (shared vertices, shared normals).
        assert(reloaded.NumVertices() == orig.NumVertices() && "OBJ vertex count mismatch");
        assert(reloaded.NumFaces() == orig.NumFaces() && "OBJ face count mismatch");

        const double tol = 1e-6;
        for (size_t i = 0; i < orig.NumVertices(); ++i)
            assert(near3(reloaded.vertices[i], orig.vertices[i], tol) &&
                   "OBJ vertex position mismatch");
        for (size_t i = 0; i < orig.NumFaces(); ++i)
            for (int k = 0; k < 3; ++k)
                assert(reloaded.faces[i][k] == orig.faces[i][k] && "OBJ face index mismatch");
        std::cout << "  ✓ OBJ write/read round-trip OK\n";
    }

    // -----------------------------------------------------------------------
    // 13. OBJ multi-mesh write
    // -----------------------------------------------------------------------
    std::cout << "\n[13] OBJ multi-mesh write" << std::endl;
    {
        const std::string obj_file = "/tmp/test_multi_mesh.obj";
        mophi::SurfaceMesh tet = make_unit_tet();
        mophi::SurfaceMesh cube = make_unit_cube();
        WriteOBJ(obj_file, std::vector<mophi::SurfaceMesh>{tet, cube});

        mophi::SurfaceMesh combined;
        assert(LoadOBJ(obj_file, combined, /*load_normals=*/false));
        const size_t expected_verts = tet.NumVertices() + cube.NumVertices();
        const size_t expected_faces = tet.NumFaces() + cube.NumFaces();
        assert(combined.NumVertices() == expected_verts && "Multi-OBJ vertex count mismatch");
        assert(combined.NumFaces() == expected_faces && "Multi-OBJ face count mismatch");
        std::cout << "  ✓ OBJ multi-mesh write/read OK ("
                  << combined.NumVertices() << " verts, " << combined.NumFaces() << " faces)\n";
    }

    // -----------------------------------------------------------------------
    // 14. ComputeMassProperties – consistency between STL and PLY loads
    // -----------------------------------------------------------------------
    std::cout << "\n[14] Mass properties consistency across file formats" << std::endl;
    {
        mophi::SurfaceMesh tet_orig = make_unit_tet();
        double vol_ref;
        mophi::Real3d ctr_ref, iner_ref, prod_ref;
        tet_orig.ComputeMassProperties(vol_ref, ctr_ref, iner_ref, prod_ref);

        // Write STL, reload, check mass props (vertex welding makes it solid)
        const std::string stl_f = "/tmp/test_mass_stl.stl";
        WriteSTL(stl_f, tet_orig);
        mophi::SurfaceMesh m_stl;
        LoadSTL(stl_f, m_stl);
        double vol_stl;
        mophi::Real3d ctr_stl, iner_stl, prod_stl;
        m_stl.ComputeMassProperties(vol_stl, ctr_stl, iner_stl, prod_stl);
        // STL uses floats, so tolerance is relaxed
        assert(near(vol_stl, vol_ref, 1e-5) && "STL mass property volume mismatch");
        assert(near3(ctr_stl, ctr_ref, 1e-5) && "STL mass property CoM mismatch");

        // Write PLY, reload, check mass props
        const std::string ply_f = "/tmp/test_mass_ply.ply";
        WritePLY(ply_f, tet_orig);
        mophi::SurfaceMesh m_ply;
        LoadPLY(ply_f, m_ply);
        double vol_ply;
        mophi::Real3d ctr_ply, iner_ply, prod_ply;
        m_ply.ComputeMassProperties(vol_ply, ctr_ply, iner_ply, prod_ply);
        assert(near(vol_ply, vol_ref, 1e-5) && "PLY mass property volume mismatch");
        assert(near3(ctr_ply, ctr_ref, 1e-5) && "PLY mass property CoM mismatch");

        std::cout << "  ✓ Mass properties are consistent across STL and PLY file formats\n";
    }

    std::cout << "\n=== All SurfaceMesh I/O and utility tests passed! ===" << std::endl;
    return 0;
}
