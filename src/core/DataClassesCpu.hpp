//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_DATA_CLASSES_CPU_HPP
#define MOPHI_DATA_CLASSES_CPU_HPP

#include <limits>
#include <vector>
#include "core/DataContainerBase.hpp"

namespace mophi {

// Stores std::vector<T> objects, which are arrays that are on host only
class HostArrayContainer : public DataContainer {
  public:
    HostArrayContainer() = default;
    ~HostArrayContainer() override {
        for (auto& [key, holder] : data_) {
            holder.reset();
        }
        data_.clear();
        meta_.clear();
    }

    // Insert a std::vector<T> with a string key
    template <typename T>
    void Add(const std::string& name) {
        Insert<std::vector<T>>(name, std::make_shared<std::vector<T>>());  // forwards to base
    }

    // Get non-const access to the std::vector<T>
    template <typename T>
    std::vector<T>& GetArray(const std::string& name) {
        return Get<std::vector<T>>(name);  // forwards to base
    }

    // Const access
    template <typename T>
    const std::vector<T>& GetArray(const std::string& name) const {
        return Get<std::vector<T>>(name);
    }
};

class HostArrayRotatingPool : public RotatingDataContainer {
  public:
    // If the scanned existing array is larger than a portion of the requested size, it will be reused.
    explicit HostArrayRotatingPool(float reuseThreshold = 0.5f) : reuse_threshold_(reuseThreshold) {}
    ~HostArrayRotatingPool() override {
        for (auto& [key, holder] : data_) {
            holder.reset();
        }
        data_.clear();
        meta_.clear();
    }

    template <typename T = char>
    void Claim(const std::string& name, size_t num_elements) {
        if (Contains(name)) {
            MOPHI_ERROR("Claim failed: key already exists: " + name);
        }

        std::string best_key;
        size_t best_fit = std::numeric_limits<size_t>::max();
        double best_fit_score = 0.0;
        bool found_no_enlarge_fit = false;

        for (const auto& [key, info] : meta_) {
            if (!info.claimed && *(info.type) == typeid(std::vector<T>)) {
                const auto& candidate = *static_cast<const Holder<std::vector<T>>&>(*data_.at(key)).value;
                size_t candidate_size = candidate.size();
                // If candidate is larger than needed, we are happy to select the smallest candidate
                // If candidate is larger than a portion of the requested size, we are happy to select the closest match
                if (candidate_size >= num_elements && candidate_size < best_fit) {
                    best_key = key;
                    best_fit = candidate_size;
                    found_no_enlarge_fit = true;
                }
                if (candidate_size >= static_cast<size_t>(num_elements * reuse_threshold_) && !found_no_enlarge_fit) {
                    double fit_score = static_cast<double>(candidate_size) / num_elements;
                    if (fit_score > best_fit_score) {
                        best_fit_score = fit_score;
                        best_key = key;
                    }
                }
            }
        }

        if (!best_key.empty()) {
            data_[name] = std::move(data_[best_key]);
            data_.erase(best_key);
            meta_[name] = {&typeid(std::vector<T>), true};
            meta_.erase(best_key);

            // Resize if needed, but do not shrink the existing vector
            auto& vec = Get<std::vector<T>>(name);
            if (vec.size() < num_elements)
                vec.resize(num_elements);
        } else {
            Insert<std::vector<T>>(name, std::make_shared<std::vector<T>>(num_elements));
        }
    }

    float reuse_threshold_ = 0.5f;  // Threshold for reusing existing arrays
};

}  // namespace mophi

#endif  // MOPHI_DATA_CLASSES_CPU_HPP
