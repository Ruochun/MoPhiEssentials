//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_THREAD_MANAGER_H
#define MOPHI_THREAD_MANAGER_H

#include <atomic>
#include <condition_variable>
#include <mutex>

// class holds on to statistics related to the scheduling process
class ManagerStatistics {
  public:
    std::atomic<uint64_t> nTimesDynamicHeldBack;
    std::atomic<uint64_t> nTimesKinematicHeldBack;
    std::atomic<uint64_t> nDynamicUpdates;
    std::atomic<uint64_t> nKinematicUpdates;
    std::atomic<uint64_t> accumKinematicLagSteps;
    // std::atomic<uint64_t> nDynamicReceives;
    // std::atomic<uint64_t> nKinematicReceives;

    ManagerStatistics() noexcept {
        nTimesDynamicHeldBack = 0;
        nTimesKinematicHeldBack = 0;
        nDynamicUpdates = 0;
        nKinematicUpdates = 0;
        accumKinematicLagSteps = 0;
        // nDynamicReceives = 0;
        // nKinematicReceives = 0;
    }

    ~ManagerStatistics() {}
};

// class that will be used via an atomic object to coordinate the
// production-consumption interplay
class ThreadManager {
  public:
    // dT's
    std::atomic<int64_t> stampLastDynamicUpdateProdDate;
    std::atomic<int64_t> currentStampOfDynamic;
    std::atomic<int64_t> dynamicMaxFutureDrift;
    std::atomic<bool> dynamicDone;

    // kT's
    std::atomic<int64_t> kinematicIngredProdDateStamp;  // dT tags this when sending it to kT
    std::atomic<int64_t> kinematicMaxFutureDrift;       // kT tags this to its produce before shipping

    std::atomic<bool> dynamicOwned_Prod2ConsBuffer_isFresh;
    std::atomic<bool> kinematicOwned_Cons2ProdBuffer_isFresh;

    std::mutex dynamicOwnedBuffer_AccessCoordination;
    std::mutex kinematicOwnedBuffer_AccessCoordination;
    std::mutex kinematicCanProceed;
    std::mutex dynamicCanProceed;
    std::condition_variable cv_KinematicCanProceed;
    std::condition_variable cv_DynamicCanProceed;
    ManagerStatistics schedulingStats;

    ThreadManager() noexcept {
        // that is, let dynamic advance into future as much as it wants, if it is -1
        dynamicMaxFutureDrift = -1;
        stampLastDynamicUpdateProdDate = -1;
        kinematicIngredProdDateStamp = -1;
        currentStampOfDynamic = 0;
        dynamicDone = false;
        dynamicOwned_Prod2ConsBuffer_isFresh = false;
        kinematicOwned_Cons2ProdBuffer_isFresh = false;
    }

    ~ThreadManager() {}

    inline int64_t getStepsSinceLastUpdate() const { return currentStampOfDynamic - stampLastDynamicUpdateProdDate; }

    inline bool dynamicShouldWait() const {
        // do not hold dynamic back under the following circustances:
        // * the update frequency is negative, dynamic can drift into future
        // * the kinematic is done
        if (dynamicMaxFutureDrift < 0)
            return false;

        // The dynamic should wait if it moved too far into the future.
        // stampLastDynamicUpdateProdDate stamps the last time when dT acquires something from kT.
        // dynamicMaxFutureDrift is the max number of cycles dT can run with no new information form kT,
        // defaulting to -1 (just keep going, no waiting for kT).
        bool shouldWait =
            (currentStampOfDynamic > stampLastDynamicUpdateProdDate + (dynamicMaxFutureDrift) ? true : false);
        // Note we do have to double-wait when we do wait.
        return shouldWait;
    }
};

// Manager of the collabortation between the main thread and worker threads
class WorkerMainCoordination {
  public:
    // To help reporting to the main thread, when the main gives orders and waiting for a `done' response.
    std::atomic<int> numWorkersDone;
    int totalWorkers;  // Must set during initialization
    std::mutex mainCanProceed;
    std::condition_variable cv_mainCanProceed;

    // The following variables are used to ensure that when an instance of worker is created, a while loop that
    // spins in place is created. It does actual work only when we tell it all preparations are done and it can proceed
    // to do the next DoDynamics call.
    std::atomic<int> numWorkersReadyToStart;
    std::atomic<uint64_t> threadEpoch;  // started threads will be assigned the epoch number
    std::atomic<bool> threadShouldJoin;
    std::mutex threadStartLock;
    std::condition_variable cv_threadStartLock;

    // It's important for them to take init values
    WorkerMainCoordination(int nWorkers) noexcept {
        totalWorkers = nWorkers;
        numWorkersDone = 0;
        numWorkersReadyToStart = 0;
        threadEpoch = 0;
        threadShouldJoin = false;
    }
    ~WorkerMainCoordination() {}
};

#endif
