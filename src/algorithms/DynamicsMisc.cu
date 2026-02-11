//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithms/StaticDeviceSubroutines.h>

#include <kernels/Compression.cuh>

namespace mophi {

//---------------- Batch kernels (AoS Real3d -> compressed; compressed -> AoS Real3<TFP>) ----------------

template <typename CompT, typename TFP>
__global__ void compress_points_kernel_T(const Real3d& __restrict__ LBF,
                                         const Real3d& __restrict__ size,
                                         const Real3<TFP>* __restrict__ pts,
                                         CompT* __restrict__ out,
                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    CompT tmp{};
    CompressPoint_T<CompT, TFP>(&pts[i], LBF, size, &tmp);
    out[i] = tmp;
}

template <typename CompT, typename TFP>
__global__ void decompress_points_kernel_T(const Real3d& __restrict__ LBF,
                                           const Real3d& __restrict__ size,
                                           const CompT* __restrict__ in,
                                           Real3<TFP>* __restrict__ out,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    DecompressPoint_T<CompT, TFP>(&in[i], LBF, size, &out[i]);
}

// ---------------------------- Host caller of the compression kernels ------------------------------

void launch_xyz_to_voxel_kernel(const Real3d& LBF,
                                const Real3d& size,
                                const Real3d* xyz,
                                CompLinear3D_128Bit* voxel_pos,
                                size_t n,
                                cudaStream_t& stream) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    compress_points_kernel_T<CompLinear3D_128Bit, double>
        <<<numBlocks, blockSize, 0, stream>>>(LBF, size, xyz, voxel_pos, n);
}

// ----------------------------------- Static routines -------------------------------------------

__global__ void AssembleTetLaplaceCOO(const TetTopo* tets,
                                      const TetEdgesLocal* edges,
                                      const CompLinear3D_128Bit* voxel_pos,
                                      const CompLog3D_64Bit* vels,
                                      const SolverParams* P,
                                      const DomainInfo* domain,
                                      COOScatterView<uNodeID_t, storeData_t> out,
                                      size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // First, bring all the vertex indices needed by threads in this block into shared memory
    if (i < n) {
        // Load tet
        const TetTopo tet = tets[i];
        const TetEdgesLocal E = edges[i];
        // Decompress vertex positions
        Real3d vpos[4];
#pragma unroll
        for (int vi = 0; vi < 4; vi++) {
            DecompressPoint_T<CompLinear3D_128Bit, double>(&voxel_pos[tet.v[vi]], domain->LBF, domain->size, &vpos[vi]);
        }
        // Decompress vertex velocities
        Real3d vvel[4];
#pragma unroll
        for (int vi = 0; vi < 4; vi++) {
            DecompressLogscale_T<CompLog3D_64Bit, double>(&vels[tet.v[vi]], P->minVel, P->maxVel, &vvel[vi]);
        }

        // // Geometry
        // const Real3d A = vpos[0];
        // const Real3d B= vpos[1];
        // const Real3d C = vpos[2];
        // const Real3d D = vpos[3];

        // const Real3d e1 = B - A, e2 = C - A, e3 = D - A;
        // const Real3d c1 = e2 % e3;
        // const Real3d c2 = e3 % e1;
        // const Real3d c3 = e1 % e2;
        // const double   det  = (e1 ^ (e2 % e3));
        // const double   invDet = (double)1 / det;
        // const double   wScale = fabs(det);

        // const double kappa = 1; //// TODO: Use a function input instead

        // // Global DOF indices (P2 tet: 4 verts + 6 edges)
        // uNodeID_t gIdx[10];
        // gIdx[0] = fs.nodeBase + (uNodeID_t)T.v[0];
        // gIdx[1] = fs.nodeBase + (uNodeID_t)T.v[1];
        // gIdx[2] = fs.nodeBase + (uNodeID_t)T.v[2];
        // gIdx[3] = fs.nodeBase + (uNodeID_t)T.v[3];
        // gIdx[4] = fs.edgeBase + (uNodeID_t)E.e[0] * fs.mEdge;
        // gIdx[5] = fs.edgeBase + (uNodeID_t)E.e[1] * fs.mEdge;
        // gIdx[6] = fs.edgeBase + (uNodeID_t)E.e[2] * fs.mEdge;
        // gIdx[7] = fs.edgeBase + (uNodeID_t)E.e[3] * fs.mEdge;
        // gIdx[8] = fs.edgeBase + (uNodeID_t)E.e[4] * fs.mEdge;
        // gIdx[9] = fs.edgeBase + (uNodeID_t)E.e[5] * fs.mEdge;

        // // Local stiffness (10x10)
        // Real K[10][10];
        // #pragma unroll
        // for (int i=0;i<10;++i){ #pragma unroll
        //     for (int j=0;j<10;++j) K[i][j] = (Real)0;
        // }

        // // Quadrature
        // for (int q=0; q<B.nQ; ++q){
        //     const Real wq = B.w[q] * wScale;

        //     Real3d gP[10];
        //     #pragma unroll
        //     for (int a=0;a<10;++a){
        //         const Real3d gRef = B.gradRef[q*10 + a];
        //         gP[a] = (c1*gRef.x() + c2*gRef.y() + c3*gRef.z()) * invDet;
        //     }

        //     #pragma unroll
        //     for (int a=0;a<10;++a){
        //         #pragma unroll
        //         for (int b=0;b<10;++b){
        //             K[a][b] += kappa * wq * (gP[a] ^ gP[b]);
        //         }
        //     }
        // }

        // // Deterministic write: 100 triplets per tet
        // const unsigned long long base = (unsigned long long)eId * 100ull;
        // unsigned long long p = base;

        // #pragma unroll
        // for (int a=0;a<10;++a){
        //     const uNodeID_t row = gIdx[a];
        //     #pragma unroll
        //     for (int b=0;b<10;++b){
        //         out.row[p] = row;
        //         out.col[p] = gIdx[b];
        //         out.val[p] = K[a][b];
        //         ++p;
        //     }
        // }
    }
}

void launch_laplace_assemble(const TetTopo* tets,
                             const TetEdgesLocal* edges,
                             const CompLinear3D_128Bit* voxel_pos,
                             const CompLog3D_64Bit* vels,
                             const SolverParams* P,
                             const DomainInfo* domain,
                             COOScatterView<uNodeID_t, storeData_t> out,
                             size_t n,
                             cudaStream_t& stream) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    AssembleTetLaplaceCOO<<<numBlocks, blockSize, 0, stream>>>(tets, edges, voxel_pos, vels, P, domain, out, n);
}

}  // namespace mophi
