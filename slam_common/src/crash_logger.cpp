/**
 * @file crash_logger.cpp
 * @brief Crash Logger 实现
 */

#include "slam_common/crash_logger.hpp"

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
namespace
{
/**
 * @brief 输出 cpptrace 与 libc 的诊断信息（用于排查 capability 为 false 的原因）
 * @note 该函数仅在初始化阶段调用，非 signal-safe；日志输出使用 spdlog（英文）。
 * @return 无
 */
void LogCpptraceEnvironmentDiagnostics()
{
    spdlog::error("cpptrace can_signal_safe_unwind: {}", cpptrace::can_signal_safe_unwind());
    spdlog::error("cpptrace can_get_safe_object_frame: {}", cpptrace::can_get_safe_object_frame());

#if defined(__GLIBC__)
    spdlog::error("Compile-time glibc macros: {}.{}", __GLIBC__, __GLIBC_MINOR__);
#else
    spdlog::error("Compile-time libc: non-glibc (or glibc macros unavailable)");
#endif

#ifdef _CS_GNU_LIBC_VERSION
    char libc_version_buf[256] = {0};
    const std::size_t libc_version_len = confstr(_CS_GNU_LIBC_VERSION, libc_version_buf, sizeof(libc_version_buf));
    if (libc_version_len > 0 && libc_version_len <= sizeof(libc_version_buf)) {
        spdlog::error("Runtime libc: {}", libc_version_buf);
    } else {
        spdlog::error("Runtime libc: unknown (confstr(_CS_GNU_LIBC_VERSION) failed)");
    }
#else
    spdlog::error("Runtime libc: _CS_GNU_LIBC_VERSION unavailable");
#endif

    spdlog::error("Hint: cpptrace::get_safe_object_frame requires _dl_find_object (glibc >= 2.35)");
}

}  // namespace

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
 * @brief CrashLogger 的私有实现
 */
class CrashLogger::Impl
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
        // 检查系统支持（不满足则降级运行：继续安装信号处理器，但崩溃堆栈信息可能不完整）
        const bool can_signal_safe_unwind = cpptrace::can_signal_safe_unwind();
        const bool can_get_safe_object_frame = cpptrace::can_get_safe_object_frame();
        if (!can_signal_safe_unwind || !can_get_safe_object_frame) {
            spdlog::error("Limited cpptrace capability detected, crash trace may be incomplete");
            LogCpptraceEnvironmentDiagnostics();
        }

        // 预热 cpptrace
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
        spdlog_logger_->info("Crash Logger initialized successfully");

        return true;
    }

    void shutdown()
    {
        if (!logger_initialized_) return;

        g_crash_logger_initialized.store(false);

        if (spdlog_logger_) {
            spdlog_logger_->info("Crash Logger shutting down");
            spdlog_logger_->flush();
        }
        spdlog::shutdown();

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

// CrashLogger 实现
CrashLogger::CrashLogger(const std::shared_ptr<spdlog::logger>& spdlog_logger) : pImpl_(std::make_unique<Impl>(spdlog_logger)) {}

CrashLogger::~CrashLogger() = default;

bool CrashLogger::initialize()
{
    return pImpl_->initialize();
}

void CrashLogger::shutdown()
{
    pImpl_->shutdown();
}

bool CrashLogger::check_system_support()
{
    return cpptrace::can_signal_safe_unwind() && cpptrace::can_get_safe_object_frame();
}

std::shared_ptr<void> CrashLogger::get_logger() const
{
    return std::static_pointer_cast<void>(pImpl_->get_logger());
}

// GlobalCrashLogger 实现
std::unique_ptr<CrashLogger> GlobalCrashLogger::instance_;

bool GlobalCrashLogger::initialize(const std::shared_ptr<spdlog::logger>& spdlog_logger)
{
    if (instance_) return true;

    instance_ = std::make_unique<CrashLogger>(spdlog_logger);
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
