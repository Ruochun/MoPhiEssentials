//  Copyright (c) 2025, Ruochun Zhang
//  SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <core/DataClasses.hpp>

using namespace mophi;

int main() {
    DeviceArrayRotatingPool pool;

    std::cout << "\n--- Initial Claims ---\n";
    pool.Claim<float>("vec1", 1000);
    pool.Claim<int>("ivec1", 500);
    assert(pool.Contains("vec1"));
    assert(pool.Contains("ivec1"));

    auto& a = pool.Get<DeviceArray<float>>("vec1");
    auto& b = pool.Get<DeviceArray<int>>("ivec1");
    assert(a.size() == 1000);
    assert(b.size() == 500);

    std::cout << "\n--- Unclaiming vec1 ---\n";
    pool.Unclaim("vec1");
    assert(!pool.IsClaimed("vec1"));

    std::cout << "\n--- Claiming new vec2 (should reuse vec1) ---\n";
    pool.Claim<float>("vec2", 500);  // should reuse vec1
    assert(pool.Contains("vec2"));
    auto& c = pool.Get<DeviceArray<float>>("vec2");
    assert(c.size() == 1000);  // still has 1000 elements from recycled vec1

    std::cout << "\n--- Final Status ---\n";
    pool.PrintStatus();

    std::cout << "\nAll tests passed.\n";
}