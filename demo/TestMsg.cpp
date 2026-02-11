//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <cstdlib>
#include <core/Logger.hpp>

using namespace mophi;

void simulateStep(int t) {
    MOPHI_INFO("Simulating step %d", t);

    if (t == 5)
        MOPHI_WARNING("Unstable behavior at step %d", t);

    if (t == 7)
        MOPHI_STATUS("fluid_loss", "Fluid loss threshold exceeded at step %d", t);

    if (t == 9)
        MOPHI_STATUS("fluid_loss", "Fluid loss exceeded again at step %d", t);  // Should overwrite

    if (t == 10)
        MOPHI_ERROR_NOTHROW("Crash imminent at step %d", t);
}

void InitSolver(double dt) {
    if (dt <= 0.0) {
        MOPHI_ERROR("Invalid timestep: %.4f", dt);
    }
    MOPHI_INFO("Solver initialized with dt = %.4f", dt);
}

int main() {
    std::cout << "=== Logger Test Script ===\n";

    Logger::GetInstance().SetVerbosity(VERBOSITY_WARNING);

    std::cout << "\n-- Begin Simulation --\n";
    for (int t = 0; t <= 10; ++t) {
        simulateStep(t);
    }

    std::cout << "\n-- Check Errors/Warnings --\n";
    if (Logger::GetInstance().HasWarnings()) {
        std::cout << "Warnings detected!\n";
    }

    if (Logger::GetInstance().HasErrors()) {
        std::cerr << "Errors detected! Printing summary:\n";
        Logger::GetInstance().PrintWarningsAndErrors();
    }

    std::cout << "\n-- All Messages (for review) --\n";
    Logger::GetInstance().PrintAll();

    std::cout << "\n-- Status Messages (deduplicated) --\n";
    Logger::GetInstance().PrintStatusMessages();

    std::cout << "\n-- Test MOPHI_ERROR --\n";
    try {
        InitSolver(0.0);
    } catch (const SolverException& ex) {
        std::cerr << "Caught exception from MOPHI_ERROR: " << ex.what() << std::endl;
    }

    std::cout << "\n-- Done --\n";

    return 0;
}
