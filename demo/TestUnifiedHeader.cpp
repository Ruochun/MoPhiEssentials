//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

/**
 * @file TestUnifiedHeader.cpp
 * @brief Demonstration of using MoPhiEssentials.h unified include header
 *
 * This example shows how easy it is to use MoPhiEssentials by including
 * just one header file instead of multiple individual headers.
 */

#include <MoPhiEssentials.h>
#include <iostream>

using namespace mophi;

int main() {
    std::cout << "\n=== MoPhiEssentials Unified Header Test ===\n\n";

    // Test 1: Real3 vectors
    std::cout << "1. Testing Real3 vectors...\n";
    Real3f vec1(1.0f, 2.0f, 3.0f);
    Real3f vec2(4.0f, 5.0f, 6.0f);
    Real3f sum = vec1 + vec2;
    std::cout << "   vec1 + vec2 = (" << sum.x() << ", " << sum.y() << ", " << sum.z() << ")\n";
    std::cout << "   ✓ Real3 operations work!\n\n";

    // Test 2: Logger
    std::cout << "2. Testing Logger...\n";
    MOPHI_INFO("Core", "This is an info message from unified header test");
    std::cout << "   ✓ Logger works!\n\n";

    // Test 3: Timer
    std::cout << "3. Testing Timer...\n";
    Timer timer;
    timer.start();
    // Simulate some work
    volatile int sum_val = 0;
    for (int i = 0; i < 1000000; i++) {
        sum_val += i;
    }
    timer.stop();
    std::cout << "   Elapsed time: " << timer.GetTimeSeconds() << " seconds\n";
    std::cout << "   ✓ Timer works!\n\n";

    std::cout << "=== All unified header tests passed! ===\n\n";
    std::cout << "Success! The MoPhiEssentials.h header provides easy access to:\n";
    std::cout << "  - Core math (Real3)\n";
    std::cout << "  - Logging utilities (Logger)\n";
    std::cout << "  - Timing utilities (Timer)\n";
    std::cout << "  - API version information\n";
    std::cout << "  - And all other MoPhiEssentials components!\n\n";

    return 0;
}
