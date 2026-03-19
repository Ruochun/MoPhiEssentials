//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_DATA_CONTAINER_BASE_HPP
#define MOPHI_DATA_CONTAINER_BASE_HPP

#include <typeinfo>
#include "Logger.hpp"

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

}  // namespace mophi

#endif
