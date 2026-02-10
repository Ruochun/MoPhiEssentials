#ifndef MOPHI_UTILS_LOGGER_H
#define MOPHI_UTILS_LOGGER_H

#include <string>
#include <iostream>

namespace MoPhi {
namespace Utils {

/**
 * @brief Simple logging utility
 * 
 * This is a placeholder. Actual implementation should be copied from MoPhi.
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void setLevel(LogLevel level) { level_ = level; }
    
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }

private:
    Logger() : level_(LogLevel::INFO) {}
    LogLevel level_;
    
    void log(LogLevel level, const std::string& msg) {
        if (level >= level_) {
            std::cout << "[" << levelToString(level) << "] " << msg << std::endl;
        }
    }
    
    std::string levelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

// Convenience macros
#define MOPHI_LOG_DEBUG(msg) MoPhi::Utils::Logger::instance().debug(msg)
#define MOPHI_LOG_INFO(msg) MoPhi::Utils::Logger::instance().info(msg)
#define MOPHI_LOG_WARNING(msg) MoPhi::Utils::Logger::instance().warning(msg)
#define MOPHI_LOG_ERROR(msg) MoPhi::Utils::Logger::instance().error(msg)

} // namespace Utils
} // namespace MoPhi

#endif // MOPHI_UTILS_LOGGER_H
