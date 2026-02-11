//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_DATA_CLASSES_HPP
#define MOPHI_DATA_CLASSES_HPP

#include <core/DataMigrationHelper.hpp>

namespace mophi {

// -------------------------------------------
// Data containers, usually have long lifetimes
// -------------------------------------------

// General-purpose data container that can hold any type of data, indexed by string keys.
class DataContainer {
  protected:
    struct IHolder {
        virtual ~IHolder() = default;
    };

    template <typename T>
    struct Holder : IHolder {
        explicit Holder(std::shared_ptr<T> v) : value(std::move(v)) {}
        std::shared_ptr<T> value;
    };

  public:
    virtual ~DataContainer() = default;

    // Insert an object of arbitrary type T, wrapped in Holder<T>
    template <typename T>
    void Insert(const std::string& key, T object) {
        if (Contains(key))
            MOPHI_ERROR("Key already exists: " + key);
        data_[key] = std::make_shared<Holder<T>>(std::move(object));
        meta_[key] = {&typeid(T), true};
    }

    template <typename T>
    void Insert(const std::string& key, const std::shared_ptr<T>& object) {
        if (Contains(key))
            MOPHI_ERROR("Key already exists: " + key);
        data_[key] = std::make_shared<Holder<T>>(std::move(object));
        meta_[key] = {&typeid(T), true};
    }

    // Get mutable reference to object of type T
    template <typename T>
    T& Get(const std::string& key) {
        check_type<T>(key);
        return *static_cast<Holder<T>&>(*data_.at(key)).value;
    }

    // Get const reference to object of type T
    template <typename T>
    const T& Get(const std::string& key) const {
        check_type<T>(key);
        return *static_cast<const Holder<T>&>(*data_.at(key)).value;
    }

    bool Contains(const std::string& key) const { return data_.count(key) > 0; }

    const std::type_info& TypeOf(const std::string& key) const {
        if (!Contains(key))
            MOPHI_ERROR("Data container key not found: " + key);
        return *meta_.at(key).type;
    }

    std::vector<std::string> Keys() const {
        std::vector<std::string> out;
        for (const auto& kv : data_)
            out.push_back(kv.first);
        return out;
    }

    void PrintSummary() const {
        for (const auto& [key, info] : meta_) {
            std::cout << "[" << key << "] "
                      << "type = " << info.type->name() << ", claimed = " << (info.claimed ? "true" : "false")
                      << std::endl;
        }
    }

  protected:
    struct Meta {
        const std::type_info* type;
        bool claimed;
    };

    template <typename T>
    void check_type(const std::string& key) const {
        if (!Contains(key))
            on_missing_key(key);
        if (*meta_.at(key).type != typeid(T))
            MOPHI_ERROR("Type mismatch for key: " + key);
    }

    virtual void on_missing_key(const std::string& key) const {
        MOPHI_ERROR("Data container key not found: '" + key + "'");
    }

    std::unordered_map<std::string, std::shared_ptr<IHolder>> data_;
    std::unordered_map<std::string, Meta> meta_;
};

// Stores DualArray<T> objects, which are arrays that are pinned to host memory and have a device counterpart
class DualArrayContainer : public DataContainer {
  public:
    DualArrayContainer() = default;
    ~DualArrayContainer() override {
        for (auto& [key, holder] : data_) {
            holder.reset();
        }
        data_.clear();
        meta_.clear();
    }

    // Insert a DualArray<T> of given size with a string key
    template <typename T>
    void Add(const std::string& name) {
        Insert<DualArray<T>>(name, std::make_shared<DualArray<T>>());  // forwards to base
    }

    // Get non-const access to the DualArray<T>
    template <typename T>
    DualArray<T>& GetArray(const std::string& name) {
        return Get<DualArray<T>>(name);  // forwards to base
    }

    // Const access
    template <typename T>
    const DualArray<T>& GetArray(const std::string& name) const {
        return Get<DualArray<T>>(name);
    }
};

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

    // Insert a std::vector<T> of given size with a string key
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

// -------------------------------------------
// Rotating data dispatchers, usually have shorter lifetimes
// -------------------------------------------

class RotatingDataContainer : public DataContainer {
  public:
    virtual ~RotatingDataContainer() = default;

    virtual void PrintStatus() const {
        for (const auto& [key, info] : meta_) {
            std::cout << "[" << key << "] " << (info.claimed ? "CLAIMED" : "IDLE") << ", type = " << info.type->name()
                      << std::endl;
        }
    }

    void Unclaim(const std::string& key) {
        if (!Contains(key)) {
            MOPHI_ERROR("Data container unclaim failed: Data container key not found: " + key);
        }
        meta_.at(key).claimed = false;
    }

    bool IsClaimed(const std::string& key) const {
        if (!Contains(key))
            return false;
        return meta_.at(key).claimed;
    }

    std::vector<std::string> IdleKeys() const {
        std::vector<std::string> out;
        for (const auto& [key, info] : meta_) {
            if (!info.claimed)
                out.push_back(key);
        }
        return out;
    }

    // To be overridden by subclasses
    template <typename T>
    void Claim(const std::string& name, std::size_t num_elements) {
        MOPHI_ERROR(std::string("Claim<T>() not implemented for this RotatingDataContainer subclass."));
    }
};

class DeviceArrayRotatingPool : public RotatingDataContainer {
  public:
    // If the scanned existing array is larger than a portion of the requested size, it will be reused.
    explicit DeviceArrayRotatingPool(float reuseThreshold = 0.5f) : reuse_threshold_(reuseThreshold) {}
    ~DeviceArrayRotatingPool() override {
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
            if (!info.claimed && *(info.type) == typeid(DeviceArray<T>)) {
                const auto& candidate = *static_cast<const Holder<DeviceArray<T>>&>(*data_.at(key)).value;
                size_t candidate_size = candidate.size();
                // If candidate is larger than needed, we are happy to select the smallest candidate
                // If candidate is larger than a portion of the requested size, we are happy to select the matchest size
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
            meta_[name] = {&typeid(DeviceArray<T>), true};
            meta_.erase(best_key);

            // This resize will not shrink the existing array, and will do nothing if the existing array is large enough
            Get<DeviceArray<T>>(name).resize(num_elements);
        } else {
            Insert<DeviceArray<T>>(name, std::make_shared<DeviceArray<T>>(num_elements));
        }
    }

    float reuse_threshold_ = 0.5f;  // Threshold for reusing existing arrays
};

// MoPhiScratchData mainly contains space allocated as system scratch pad and as thread temporary arrays
class MoPhiScratchData {
  private:
  public:
    char* allocateScratchSpace(size_t size) { return nullptr; }
};

}  // namespace mophi

#endif
