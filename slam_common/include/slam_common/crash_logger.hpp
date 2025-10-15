/**
 * @file crash_logger.hpp
 * @brief SLAM 崩溃日志记录器，集成 spdlog 的信号安全崩溃处理
 *
 * 该模块提供信号安全的崩溃处理方案，当程序异常退出时
 * 自动将 cpptrace 堆栈信息写入 spdlog 日志文件。
 */

#pragma once

#include <string>
#include <memory>
#include <functional>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/dup_filter_sink.h>

namespace ms_slam::slam_common
{
/**
 * @brief 日志记录器配置
 */
struct LoggerConfig {
    /// spdlog日志文件路径
    std::string log_file_path = "slam.log";

    /// 临时跟踪文件目录
    std::string temp_dir = "/tmp";

    /// 日志级别（spdlog格式）
    std::string log_level = "info";

    /// 日志刷新级别（spdlog格式）
    std::string flush_level = "warn";

    /// 是否启用文件轮转
    bool enable_rotation = false;

    /// 轮转文件大小（MB）
    size_t max_file_size_mb = 10;

    /// 保留的轮转文件数量
    size_t max_files = 5;

    /// 自定义日志格式
    std::string log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [SLAM] %v";
};

/**
 * @brief SLAM 崩溃日志记录器
 */
class CrashLogger
{
  public:
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit CrashLogger(const std::shared_ptr<spdlog::logger>& spdlog_logger);

    /**
     * @brief 析构函数
     */
    ~CrashLogger();

    // 禁止复制和移动
    CrashLogger(const CrashLogger&) = delete;
    CrashLogger& operator=(const CrashLogger&) = delete;
    CrashLogger(CrashLogger&&) = delete;
    CrashLogger& operator=(CrashLogger&&) = delete;

    /**
     * @brief 初始化崩溃日志记录器
     * @return 初始化是否成功
     */
    bool initialize();

    /**
     * @brief 停止崩溃日志记录器
     */
    void shutdown();

    /**
     * @brief 检查系统是否支持signal-safe tracing
     * @return 是否支持
     */
    static bool check_system_support();

    /**
     * @brief 获取spdlog logger实例
     * @return 共享的logger指针，可用于正常日志记录
     */
    std::shared_ptr<void> get_logger() const;

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * @brief 全局崩溃日志记录器实例
 *
 * 这是一个便利的全局实例，用于简化使用。
 * 在程序开始时调用 InitializeGlobalCrashLogger()，
 * 程序结束时会自动清理。
 */
class GlobalCrashLogger
{
  public:
    /**
     * @brief 初始化全局崩溃日志记录器
     * @param config 配置参数
     * @return 初始化是否成功
     */
    static bool initialize(const std::shared_ptr<spdlog::logger>& spdlog_logger);

    /**
     * @brief 获取全局logger实例
     * @return 共享的logger指针
     */
    static std::shared_ptr<void> get_logger();

    /**
     * @brief 关闭全局崩溃日志记录器
     */
    static void shutdown();

    /**
     * @brief 检查全局实例是否已初始化
     */
    static bool is_initialized();

  private:
    static std::unique_ptr<CrashLogger> instance_;
};

}  // namespace ms_slam::slam_common

// 便利宏定义
#define SLAM_CRASH_LOGGER_INIT(spdlog_logger) ms_slam::slam_common::GlobalCrashLogger::initialize(spdlog_logger)

#define SLAM_CRASH_LOGGER_SHUTDOWN() ms_slam::slam_common::GlobalCrashLogger::shutdown()
