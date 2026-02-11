//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "GpuManager.h"
#include "Logger.hpp"

namespace mophi {

GpuManager::GpuManager(unsigned int total_streams, const std::vector<int>& device_ids) {
    ndevices = device_ids.size();
    nstreams = total_streams;
    // Actual device IDs that are accessible
    int actual_ndevices = ScanNumDevices();

    // streams stores, using the device index, the streams available for each device
    this->streams.resize(ndevices);

    for (unsigned int current_device_idx = 0; total_streams > 0; total_streams--, current_device_idx++) {
        if (current_device_idx >= ndevices) {
            current_device_idx = 0;
        }
        int this_device_id = device_ids[current_device_idx];
        if (this_device_id >= actual_ndevices) {
            MOPHI_ERROR("Device ID %d is not available! Available devices: %d", this_device_id, actual_ndevices);
        }

        // Stream is created later with the worker threads
        cudaStream_t new_stream = nullptr;

        this->streams[this_device_id].push_back(StreamInfo{this_device_id, new_stream, false});
    }
}

GpuManager::~GpuManager() {
    for (auto outer = this->streams.begin(); outer != this->streams.end(); outer++) {
        for (auto stream = outer->begin(); stream != outer->end(); stream++) {
            if (stream->stream != nullptr) {
                MOPHI_GPU_CALL_NOTHROW(cudaStreamDestroy(stream->stream));
            }
        }
    }
}

int GpuManager::ScanNumDevices() {
    int ndevices = 0;
    MOPHI_GPU_CALL(cudaGetDeviceCount(&ndevices));
    return ndevices;
}

unsigned int GpuManager::GetStreamsPerDevice() {
    auto iter = std::min_element(this->streams.begin(), this->streams.end(),
                                 [](const auto& a, const auto& b) { return (a.size() < b.size()); });
    return (*iter).size();
}

unsigned int GpuManager::GetMaxStreamsPerDevice() {
    auto iter = std::max_element(this->streams.begin(), this->streams.end(),
                                 [](const auto& a, const auto& b) { return (a.size() < b.size()); });
    return (*iter).size();
}

const std::vector<GpuManager::StreamInfo>& GpuManager::GetStreamsFromDevice(int index) {
    return this->streams[index];
}

const std::vector<GpuManager::StreamInfo>& GpuManager::GetStreamsFromDevice(const GpuManager::StreamInfo& info) {
    return this->GetStreamsFromDevice(info.device);
}

const GpuManager::StreamInfo& GpuManager::GetAvailableStream() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Iterate over stream lists by device
    for (auto by_device = this->streams.begin(); by_device != streams.end(); by_device++) {
        // Iterate over streams in each device
        for (auto stream = by_device->begin(); stream != by_device->end(); stream++) {
            if (!stream->_impl_active) {
                stream->_impl_active = true;
                return *stream;
            }
        }
    }

    // This exception is not meant to be handled, it serves as a notifier that the algorithm is using more streams than
    // it allocated
    MOPHI_ERROR("No available streams. In total, " + std::to_string(nstreams) + " streams were requested!");
}

const GpuManager::StreamInfo& GpuManager::GetAvailableStreamFromDevice(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto stream = this->streams[index].begin(); stream != this->streams[index].end(); stream++) {
        if (!stream->_impl_active) {
            stream->_impl_active = true;
            return *stream;
        }
    }

    // This exception should rarely be thrown, so it shouldn't have a notable performance impact
    MOPHI_ERROR("No available streams on device [" + std::to_string(index) + "]!");
}

void GpuManager::SetStreamAvailable(const StreamInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto stream = this->streams[info.device].begin(); stream != this->streams[info.device].end(); stream++) {
        if (stream->stream == info.stream) {
            stream->_impl_active = false;
        }
    }
}

}  // namespace mophi
