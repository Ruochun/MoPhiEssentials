//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <thread>
#include <chrono>
#include <utils/Timer.hpp>

int main() {
    mophi::Timer timer;

    // Start the timer
    timer.start();

    // Simulate some work with a sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Stop the timer
    timer.stop();

    // Print the elapsed time
    std::cout << "Elapsed time: " << timer.GetTimeSeconds() << " seconds" << std::endl;

    return 0;
}