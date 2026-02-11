//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_GPU_MANAGER_H
#define MOPHI_GPU_MANAGER_H

#include <cuda_runtime_api.h>
#include <vector>
#include <mutex>
#include <core/ApiVersion.h>

namespace mophi {

class GpuManager {
  public:
    // Constructor: trys to bind this number of streams to the devices specified in device_ids. If more streams than the
    // device numbers given, then it goes back to the start of device_ids and start over.
    GpuManager(unsigned int total_streams, const std::vector<int>& device_ids);
    ~GpuManager();

    struct StreamInfo {
      public:
        int device;
        cudaStream_t stream;

        bool _impl_active;  // Reserved for the implementation
    };

    // Returns the LEAST number of streams available on any device.
    unsigned int GetStreamsPerDevice();
    // Returns the HIGHEST number of streams per device.
    unsigned int GetMaxStreamsPerDevice();

    static int ScanNumDevices();

    // DO NOT USE UNLESS YOU INTEND TO MANUALLY HANDLE YOUR STREAMS.
    const std::vector<StreamInfo>& GetStreamsFromDevice(int index);

    // DO NOT USE UNLESS YOU INTEND TO MANUALLY HANDLE YOUR STREAMS.
    const std::vector<StreamInfo>& GetStreamsFromDevice(const StreamInfo&);

    // Get a stream which hasn't been used yet and mark it as used.
    const StreamInfo& GetAvailableStream();
    const StreamInfo& GetAvailableStreamFromDevice(int index);

    // Mark a stream as unused.
    void SetStreamAvailable(const StreamInfo&);

    // Return the number of devices detected.
    int GetNumDevices() { return ndevices; }

  private:
    int ndevices;
    int nstreams;
    std::vector<std::vector<StreamInfo>> streams;

    mutable std::mutex mutex_;
};

}  // namespace mophi

#endif
