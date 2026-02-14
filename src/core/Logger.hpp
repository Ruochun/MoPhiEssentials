//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_GPU_ERROR_H
#define MOPHI_GPU_ERROR_H

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

#include <cuda_runtime_api.h>
#include <core/BaseClasses.hpp>
#include <common/Defines.hpp>

namespace mophi {

// -----------------------------
// Logging types and structure
// -----------------------------

enum class MessageType { Info, Warning, Error, Status };

struct LogMessage {
    MessageType type;
    std::string source;
    std::string message;
    std::string file;
    int line;
    std::string identifier;
};

// -----------------------------
// Logger and exception class (thread-safe, singleton)
// -----------------------------

class SolverException : public std::runtime_error {
  public:
    SolverException(const std::string& msg) : std::runtime_error(msg) {}
};

class Logger : private NonCopyable, public Singleton<Logger> {
  public:
    void SetVerbosity(verbosity_t level) {
        std::lock_guard<std::mutex> lock(mutex_);
        verbosity = level;
    }

    void Log(MessageType type, const std::string& source, const std::string& msg, const std::string& file, int line) {
        std::lock_guard<std::mutex> lock(mutex_);
        LogMessage Log{type, source, msg, file, line};
        logs.push_back(Log);

        if (should_print_immediately(type)) {
            std::cout << format(Log) << std::endl;
        }
    }

    // snprintf version of logging with format arguments
    template <typename... Args>
    std::string Logf(MessageType type, const char* func, const char* file, int line, const char* fmt, Args&&... args) {
        constexpr size_t BUF_SIZE = 2048;
        char buffer[BUF_SIZE];
        std::snprintf(buffer, BUF_SIZE, fmt, std::forward<Args>(args)...);
        std::string message(buffer);
        Log(type, func, message, file, line);
        return message;
    }

    // const char* version without format arguments (avoids -Wformat-security warning)
    std::string Logf(MessageType type, const char* func, const char* file, int line, const char* message) {
        Log(type, func, std::string(message), file, line);
        return std::string(message);
    }

    // std::string version of logging
    std::string Logf(MessageType type, const char* func, const char* file, int line, const std::string& message) {
        Log(type, func, message, file, line);
        return message;
    }

    // LogStatus with format arguments
    template <typename... Args>
    void LogStatusf(const std::string& identifier,
                    const char* func,
                    const char* file,
                    int line,
                    const char* fmt,
                    Args&&... args) {
        constexpr size_t BUF_SIZE = 2048;
        char buffer[BUF_SIZE];
        std::snprintf(buffer, BUF_SIZE, fmt, std::forward<Args>(args)...);
        LogStatus(identifier, func, std::string(buffer), file, line);
    }

    // LogStatus without format arguments (avoids -Wformat-security warning)
    void LogStatusf(const std::string& identifier,
                    const char* func,
                    const char* file,
                    int line,
                    const char* message) {
        LogStatus(identifier, func, std::string(message), file, line);
    }

    void LogStatus(const std::string& identifier,
                   const std::string& func,
                   const std::string& msg,
                   const std::string& file,
                   int line) {
        std::lock_guard<std::mutex> lock(mutex_);

        LogMessage Log{MessageType::Status, func, msg, file, line, identifier};
        status_messages[identifier] = Log;

        if (should_print_immediately(MessageType::Status)) {
            std::cout << format(Log) << std::endl;
        }
    }

    void PrintAll(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            os << format(Log) << std::endl;
        }
    }

    void PrintWarningsAndErrors(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Warning || Log.type == MessageType::Error) {
                os << format(Log) << std::endl;
            }
        }
    }

    void PrintStatusMessages(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : status_messages) {
            os << format(pair.second) << std::endl;
        }
    }

    bool HasErrors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Error)
                return true;
        }
        return false;
    }

    bool HasWarnings() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Warning)
                return true;
        }
        return false;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        logs.clear();
        status_messages.clear();
    }

  private:
    friend class Singleton<Logger>;  // Allows Singleton base to construct it

    Logger() : verbosity(VERBOSITY_WARNING) {}
    ~Logger() {}

    bool should_print_immediately(MessageType type) const {
        switch (type) {
            case MessageType::Error:
                return verbosity >= VERBOSITY_ERROR;
            case MessageType::Warning:
                return verbosity >= VERBOSITY_WARNING;
            case MessageType::Info:
                return verbosity >= VERBOSITY_INFO;
            // Debug level will just immediately output the previous three
            case MessageType::Status:
                return verbosity >= VERBOSITY_STEP_DEBUG;
            default:
                return false;
        }
    }

    std::string format(const LogMessage& Log) const {
        std::ostringstream oss;
        switch (Log.type) {
            case MessageType::Error:
                oss << "[ERROR]   ";
                break;
            case MessageType::Warning:
                oss << "[WARNING] ";
                break;
            case MessageType::Info:
                oss << "[INFO]    ";
                break;
            case MessageType::Status:
                oss << "[STATUS]  ";
                break;
        }
        oss << Log.source << ": " << Log.message << " (" << Log.file << ":" << Log.line << ")";
        if (!Log.identifier.empty() && Log.type == MessageType::Status)
            oss << " [id: " << Log.identifier << "]";
        return oss.str();
    }

    mutable std::mutex mutex_;
    std::vector<LogMessage> logs;
    std::unordered_map<std::string, LogMessage> status_messages;
    verbosity_t verbosity;
};

// -----------------------------
// Logging utils for easy usage
// -----------------------------

#define MOPHI_ERROR(...)          \
    throw mophi::SolverException( \
        mophi::Logger::GetInstance().Logf(mophi::MessageType::Error, __func__, __FILE__, __LINE__, __VA_ARGS__))

#define MOPHI_ERROR_NOTHROW(...) \
    mophi::Logger::GetInstance().Logf(mophi::MessageType::Error, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define MOPHI_WARNING(...) \
    mophi::Logger::GetInstance().Logf(mophi::MessageType::Warning, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define MOPHI_INFO(...) \
    mophi::Logger::GetInstance().Logf(mophi::MessageType::Info, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define MOPHI_STATUS(identifier, ...) \
    mophi::Logger::GetInstance().LogStatusf(identifier, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define MOPHI_GPU_CALL(code)                                       \
    {                                                              \
        cudaError_t res = (code);                                  \
        if (res != cudaSuccess) {                                  \
            MOPHI_ERROR("GPU Error: %s", cudaGetErrorString(res)); \
        }                                                          \
    }

#define MOPHI_GPU_CALL_NOTHROW(code)                                       \
    {                                                                      \
        cudaError_t res = (code);                                          \
        if (res != cudaSuccess) {                                          \
            MOPHI_ERROR_NOTHROW("GPU Error: %s", cudaGetErrorString(res)); \
        }                                                                  \
    }

inline std::string pretty_format_bytes(size_t bytes) {
    // set up byte prefixes
    constexpr size_t KIBI = 1024;
    constexpr size_t MEBI = KIBI * KIBI;
    constexpr size_t GIBI = KIBI * KIBI * KIBI;
    float gibival = float(bytes) / GIBI;
    float mebival = float(bytes) / MEBI;
    float kibival = float(bytes) / KIBI;
    std::stringstream ret;
    if (gibival > 1) {
        ret << gibival << " GiB";
    } else if (mebival > 1) {
        ret << mebival << " MiB";
    } else if (kibival > 1) {
        ret << kibival << " KiB";
    } else {
        ret << bytes << " B";
    }
    return ret.str();
}

}  // namespace mophi

#endif