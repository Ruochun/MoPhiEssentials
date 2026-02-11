#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <core/DataMigrationHelper.hpp>
#include <utils/HostHelpers.hpp>

using namespace mophi;

// Define a simple struct to use in the DualStruct class
struct MyData {
    int a;
    float b;
    double c;

    MyData() : a(0), b(0.0f), c(0.0) {}
    MyData(int a, float b, double c) : a(a), b(b), c(c) {}
};

void testDualStruct() {
    // Create DualStruct with MyData type
    DualStruct<MyData> dual_struct;

    // Set values on host
    dual_struct->a = 42;
    dual_struct->b = 3.14f;
    dual_struct->c = 2.718;

    // Ensure the data is marked as modified on host
    std::cout << "Host data after modification: "
              << "a = " << dual_struct->a << ", "
              << "b = " << dual_struct->b << ", "
              << "c = " << dual_struct->c << std::endl;

    // Sync the data to device
    dual_struct.ToDevice();

    // Now check device data (it should be synced from the host)
    std::cout << "Device data after sync: "
              << "a = " << GetDeviceArray(&(dual_struct.device()->a))[0] << ", "
              << "b = " << GetDeviceArray(&(dual_struct.device()->b))[0] << ", "
              << "c = " << GetDeviceArray(&(dual_struct.device()->c))[0] << std::endl;

    // Modify host data again
    dual_struct->a = 99;
    dual_struct->b = 6.28f;
    dual_struct->c = 1.414;

    // Sync device data to host
    dual_struct.ToHost(true);  // Force sync from device to host

    // Check if device-to-host sync is successful
    std::cout << "Host data after device-to-host sync: "
              << "a = " << dual_struct->a << ", "
              << "b = " << dual_struct->b << ", "
              << "c = " << dual_struct->c << std::endl;
}

void testDualArray() {
    // Test the DualArray class
    DualArray<float> dualArray(10, 1.0f);

    std::cout << "Initial Array size: " << dualArray.size() << std::endl;

    // Resize and check
    dualArray.resize(20);
    std::cout << "Resized Array size: " << dualArray.size() << std::endl;

    // Set and get values
    dualArray.SetVal(3.14f, 5);
    std::cout << "Value at index 5: " << dualArray.GetVal(5) << std::endl;

    // Sync to device
    dualArray.ToDevice();
    std::cout << "Synced to device" << std::endl;

    // Sync to host
    dualArray.ToHost();
    std::cout << "Synced to host" << std::endl;

    // Test partial sync
    dualArray.ToHost(2, 3);  // Sync 3 elements from device to host starting at index 2
    std::cout << "Partial sync completed" << std::endl;

    // Test getting value from the device
    float val = dualArray.GetVal(5);
    std::cout << "Value from device at index 5: " << val << std::endl;

    // Force sync
    dualArray.SetVal_ForceSync(2.718f, 10);
    std::cout << "Force sync value set at index 10" << std::endl;
}

int main() {
    std::cout << "=== DualArray Test ===\n";
    testDualArray();
    std::cout << "=== DualStruct Test ===\n";
    testDualStruct();

    return 0;
}
