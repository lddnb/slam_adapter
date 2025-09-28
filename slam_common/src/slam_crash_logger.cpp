/**
 * @file slam_crash_logger.cpp
 * @brief SLAM Crash Logger 实现
 */

#include "slam_common/slam_crash_logger.hpp"

#include <sys/inotify.h>
#include <unistd.h>
#include <fcntl.h>
#include <atomic>
#include <fstream>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cstring>

#include <cpptrace/cpptrace.hpp>

namespace ms_slam::slam_common
{

// 全局状态变量（signal-safe）
std::atomic<bool> g_crash_logger_initialized{false};
char g_temp_dir[256] = "/tmp";
char g_crash_file_prefix[64] = "slam_crash_";

/**
 * @brief Signal-safe的崩溃处理器
 */
void signal_safe_crash_handler(int sig, siginfo_t* info, void* context)
{
    if (!g_crash_logger_initialized.load()) return;

    // 生成唯一的crash文件名
    char filename[512];
    snprintf(
        filename,
        sizeof(filename),
        "%s/%s_%ld.crash",
        g_temp_dir,
        g_crash_file_prefix,
        static_cast<long>(std::chrono::system_clock::now().time_since_epoch().count()));

    // 创建文件并写入基本信息
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) return;

    // 写入信号信息
    char header[256];
    int header_len = snprintf(header, sizeof(header), "SIGNAL:%d\nTIME:%ld\n", sig, static_cast<long>(time(nullptr)));
    write(fd, header, header_len);

    // 写入原始trace数据
    const char* section_marker = "TRACE_START\n";
    write(fd, section_marker, strlen(section_marker));

    try {
        // 生成原始trace并写入文件
        cpptrace::frame_ptr buffer[100];
        std::size_t frame_count = cpptrace::safe_generate_raw_trace(buffer, 100);

        for (std::size_t i = 0; i < frame_count; i++) {
            cpptrace::safe_object_frame frame;
            cpptrace::get_safe_object_frame(buffer[i], &frame);
            write(fd, &frame, sizeof(frame));
        }
    } catch (...) {
        // 在signal handler中不能抛出异常，写入错误标记
        const char* error_marker = "TRACE_GENERATION_FAILED\n";
        write(fd, error_marker, strlen(error_marker));
    }

    const char* end_marker = "\nTRACE_END\n";
    write(fd, end_marker, strlen(end_marker));

    close(fd);

    // 调用默认处理器
    signal(sig, SIG_DFL);
    kill(getpid(), sig);
}

/**
 * @brief SlamCrashLogger的私有实现
 */
class SlamCrashLogger::Impl
{
  public:
    explicit Impl(const std::shared_ptr<spdlog::logger>& spdlog_logger)
    : spdlog_logger_(spdlog_logger),
      logger_initialized_(false),
      temp_dir_(std::string(g_temp_dir))
    {
    }

    ~Impl() { shutdown(); }

    bool initialize()
    {
        try {
            // 检查系统支持
            if (!cpptrace::can_signal_safe_unwind() || !cpptrace::can_get_safe_object_frame()) {
                spdlog::error("System doesn't support signal-safe unwinding or safe object frame");
                return false;
            }

            // 预热cpptrace
            cpptrace::frame_ptr buffer[10];
            cpptrace::safe_generate_raw_trace(buffer, 10);
            cpptrace::safe_object_frame frame;
            cpptrace::get_safe_object_frame(buffer[0], &frame);

            // 安装信号处理器
            install_signal_handlers();

            // 设置全局状态
            strncpy(g_temp_dir, temp_dir_.c_str(), sizeof(g_temp_dir) - 1);
            g_crash_logger_initialized.store(true);

            logger_initialized_ = true;
            spdlog_logger_->info("SLAM Crash Logger initialized successfully");

            return true;
        } catch (const std::exception& e) {
            spdlog::error("Failed to initialize SlamCrashLogger: {}", e.what());
            return false;
        }
    }

    void shutdown()
    {
        if (!logger_initialized_) return;

        g_crash_logger_initialized.store(false);

        if (spdlog_logger_) {
            spdlog_logger_->info("SLAM Crash Logger shutting down");
            spdlog_logger_->flush();
        }

        logger_initialized_ = false;
    }

    std::shared_ptr<spdlog::logger> get_logger() const { return spdlog_logger_; }

  private:
    void install_signal_handlers()
    {
        struct sigaction sa;
        sa.sa_sigaction = signal_safe_crash_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO;

        // 安装常见的崩溃信号处理器
        sigaction(SIGSEGV, &sa, nullptr);  // 段错误
        sigaction(SIGABRT, &sa, nullptr);  // 中止
        sigaction(SIGFPE, &sa, nullptr);   // 浮点异常
        sigaction(SIGBUS, &sa, nullptr);   // 总线错误
        sigaction(SIGILL, &sa, nullptr);   // 非法指令
    }

  private:
    std::shared_ptr<spdlog::logger> spdlog_logger_;
    std::atomic<bool> logger_initialized_;
    std::string temp_dir_;
};

// SlamCrashLogger implementation
SlamCrashLogger::SlamCrashLogger(const std::shared_ptr<spdlog::logger>& spdlog_logger) : pImpl_(std::make_unique<Impl>(spdlog_logger)) {}

SlamCrashLogger::~SlamCrashLogger() = default;

bool SlamCrashLogger::initialize()
{
    return pImpl_->initialize();
}

void SlamCrashLogger::shutdown()
{
    pImpl_->shutdown();
}

bool SlamCrashLogger::check_system_support()
{
    return cpptrace::can_signal_safe_unwind() && cpptrace::can_get_safe_object_frame();
}

std::shared_ptr<void> SlamCrashLogger::get_logger() const
{
    return std::static_pointer_cast<void>(pImpl_->get_logger());
}

// GlobalCrashLogger implementation
std::unique_ptr<SlamCrashLogger> GlobalCrashLogger::instance_;

bool GlobalCrashLogger::initialize(const std::shared_ptr<spdlog::logger>& spdlog_logger)
{
    if (instance_) return true;

    instance_ = std::make_unique<SlamCrashLogger>(spdlog_logger);
    return instance_->initialize();
}

std::shared_ptr<void> GlobalCrashLogger::get_logger()
{
    return instance_ ? instance_->get_logger() : nullptr;
}

void GlobalCrashLogger::shutdown()
{
    if (instance_) {
        instance_->shutdown();
        instance_.reset();
    }
}

bool GlobalCrashLogger::is_initialized()
{
    return static_cast<bool>(instance_);
}

}  // namespace ms_slam::slam_common
