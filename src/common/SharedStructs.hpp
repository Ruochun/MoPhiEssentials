//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_SHARED_STRUCTS_HPP
#define MOPHI_SHARED_STRUCTS_HPP

#include <core/DataClasses.hpp>
#include <core/CudaAllocator.hpp>
#include <core/ManagedMemory.hpp>
#include <core/Logger.hpp>
#include <core/DataMigrationHelper.hpp>
#include <core/Real3.hpp>
#include <kernels/HelperKernels.cuh>
#include <common/Defines.hpp>
#include <utils/HostHelpers.hpp>

#include <sstream>
#include <exception>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <filesystem>
#include <cstring>
#include <string>
#include <cassert>
#include <typeinfo>
#include <typeindex>

namespace mophi {

// =============================================================================
// HOST-SIDE STRUCTS THAT ARE ESSENTIAL FOR THE CFD SOLVER
// =============================================================================

// What kind of layout the field uses
enum class SpaceKind : uint8_t {
    NodeLag,  // Continuous Lagrange nodal space (CG P^k)
    NodeIGA,  // Isogeometric Analysis (IGA) space, but not implemented yet
    Cell,     // Cell-centered (FV/P0, etc.)
    DGk       // Discontinuous element-local P^k (tet total-degree / hex tensor-product)
};
struct SpaceSpec {
    SpaceKind kind;
    int order = 1;  // applies to NodeLag (CG P^k), NodeIGA and DGk
    int nComp = 1;  // scalar=1, vector=3, etc.
};

// Solver main class-managed solver parameters.
struct SolverParams {
    double dt = -1.0;       ///< Time step for the solver
    uint64_t numSteps = 0;  ///< Number of steps already run in the simulation
    double t = 0.0;         ///< Current time in the simulation

    double minVel = 1e-12;      ///< Minimum velocity in the simulation, used for vel compression
    double maxVel = 1e6;        ///< Maximum velocity in the simulation, used for vel compression
    double velZeroEps = 1e-12;  ///< Vel under this threshold is considered zero
};

// The bounding box of the entire simulation domain, used to compress positional data.
struct DomainInfo {
    Real3d LBF;   ///< Left-bottom-front corner of the domain
    Real3d size;  ///< Size of the domain (max - LBF)
};

// Policies, user-flavors
struct SolverPolicies {
    bool typicalNS = false;  ///< Whether to create u, p function spaces automatically
};

// The information that is shared between the main thread and the worker threads. The main thread keeps a copy, the
// workers keep a const reference to it.
class SharedContext {
  public:
    DualStruct<SolverParams> solverParams = DualStruct<SolverParams>(SolverParams{});  // Always use default values
    DualStruct<DomainInfo> domainInfo = DualStruct<DomainInfo>(DomainInfo{});
    SolverPolicies policies = SolverPolicies{};
};

// Pre-sized scatter: kernels compute a base offset per element and write without atomics
template <typename Index, typename T>
struct COOScatterView {
    Index* row = nullptr;
    Index* col = nullptr;
    T* val = nullptr;

    __device__ inline void Write(size_t ofs, Index r, Index c, T v) {
        row[ofs] = r;
        col[ofs] = c;
        val[ofs] = v;
    }
};

// A parition (correspond to a dT)
struct Partition {};

// ---- Delayed-initialization handle ----

// An object that is returned to the user, but only useful after the solver is initialized, typically.
// For this struct, even if a copy is returned (not a pointer) to the user, it still handles received changes made to
// the managed object, as per how DeferredHandle is designed.
template <class T>
class DeferredHandle {
  public:
    using Ptr = std::shared_ptr<T>;

    DeferredHandle() : state_(std::make_shared<State>()) {}
    explicit DeferredHandle(std::string tag) : state_(std::make_shared<State>(std::move(tag))) {}

    // --- User side ---
    Ptr Get() const {
        std::lock_guard<std::mutex> lk(state_->mtx);
        if (!state_->ptr) {
            MOPHI_ERROR("This DeferredHandle is not ready: " + state_->tag +
                        ". (Did you call Initialize() before calling this?)");
        }
        return state_->ptr;
    }

    T* operator->() const { return Get().get(); }
    T& operator*() const { return *Get(); }

    explicit operator bool() const noexcept {
        std::lock_guard<std::mutex> lk(state_->mtx);
        return static_cast<bool>(state_->ptr);
    }

    // Optional helpers
    bool IsReady() const noexcept {
        std::lock_guard<std::mutex> lk(state_->mtx);
        return static_cast<bool>(state_->ptr);
    }
    const std::string& Tag() const noexcept { return state_->tag; }
    void SetTag(std::string tag) {
        std::lock_guard<std::mutex> lk(state_->mtx);
        state_->tag = std::move(tag);
    }

    // --- Solver side ---
    void Fulfill(Ptr p) const {
        std::lock_guard<std::mutex> lk(state_->mtx);
        if (state_->ptr)
            MOPHI_ERROR("DeferredHandle already fulfilled: " + state_->tag);
        state_->ptr = std::move(p);
    }

    template <class... Args>
    void Emplace(Args&&... args) const {
        Fulfill(std::make_shared<T>(std::forward<Args>(args)...));
    }

  private:
    struct State {
        explicit State(std::string t = "<unnamed handle>") : tag(std::move(t)) {}
        mutable std::mutex mtx;
        Ptr ptr;
        std::string tag;
    };
    std::shared_ptr<State> state_;  // <-- shared across copies
};

// =============================================================================
// NOW DEFINING MACRO COMMANDS USED BY THE DEM MODULE
// =============================================================================

// Jitify options include suppressing variable-not-used warnings. We could use CUDA lib functions too.
#define MOPHI_JITIFY_DEFAULT_OPTIONS                                                                   \
    {                                                                                                  \
        "-I" + (JitHelper::KERNEL_INCLUDE_DIR).string(), "-I" + (JitHelper::KERNEL_DIR).string(),      \
            "-I" + std::string(MOPHI_CUDA_TOOLKIT_HEADERS), "-diag-suppress=550", "-diag-suppress=177" \
    }

}  // namespace mophi

#endif
