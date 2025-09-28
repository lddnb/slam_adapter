#include <slam_common/slam_crash_logger.hpp>

#include <iostream>
#include <chrono>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/dup_filter_sink.h>

void trigger_segfault()
{
    std::cout << "Triggering segmentation fault in 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    int* ptr = nullptr;
    *ptr = 42;  // 这将触发SIGSEGV
}

void trigger_abort()
{
    std::cout << "Triggering abort in 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::abort();  // 这将触发SIGABRT
}

void trigger_fpe()
{
    std::cout << "Triggering floating point exception in 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    volatile int x = 1;
    volatile int y = 0;
    volatile int z = x / y;  // 这将触发SIGFPE
    (void)z;
}

void normal_function_call()
{
    std::cout << "Executing normal function call..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void deep_function_call_1()
{
    normal_function_call();
    trigger_segfault();  // 在深层调用栈中触发崩溃
}

void deep_function_call_2()
{
    deep_function_call_1();
}

void deep_function_call_3()
{
    deep_function_call_2();
}

int main(int argc, char* argv[])
{
    std::cout << "=== SLAM spdlog Crash Logger Test Example ===" << std::endl;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <crash_type>" << std::endl;
        std::cout << "Crash types:" << std::endl;
        std::cout << "  1 - Segmentation fault (simple)" << std::endl;
        std::cout << "  2 - Segmentation fault (deep stack)" << std::endl;
        std::cout << "  3 - Abort" << std::endl;
        std::cout << "  4 - Floating point exception" << std::endl;
        std::cout << "  5 - Test crash logging (no actual crash)" << std::endl;
        std::cout << "  0 - No crash (test normal execution)" << std::endl;
        return 1;
    }

    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "slam_crash.log";
    config.temp_dir = "/tmp";
    config.log_level = "info";

    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("slam_crash_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::from_str(config.log_level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);

    // 首先检查系统支持
    if (!ms_slam::slam_common::SlamCrashLogger::check_system_support()) {
        spdlog::error("❌ System doesn't support signal-safe unwinding!");
        return 1;
    }
    spdlog::info("✅ System supports signal-safe unwinding");

    spdlog::info("Starting SLAM test");

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("❌ Failed to initialize crash logger!");
        return 1;
    }
    spdlog::info("Crash logger initialized successfully");

    int crash_type = std::atoi(argv[1]);

    spdlog::info("Log file location: {}", std::filesystem::absolute(config.log_file_path).string());
    spdlog::info("Temp directory: {}", config.temp_dir);

    switch (crash_type) {
        case 0:
            spdlog::info("✨ Running normal execution (no crash)...");
            normal_function_call();
            spdlog::info("Normal execution completed successfully");

            spdlog::info("✅ Normal execution completed successfully!");
            break;

        case 1:
            spdlog::info("💥 Testing simple segmentation fault...");
            spdlog::warn("About to trigger segmentation fault");
            trigger_segfault();
            break;

        case 2:
            spdlog::info("💥 Testing segmentation fault with deep call stack...");
            spdlog::warn("About to trigger segmentation fault in deep call stack");
            deep_function_call_3();
            break;

        case 3:
            spdlog::info("💥 Testing abort...");
            spdlog::warn("About to trigger abort");
            trigger_abort();
            break;

        case 4:
            spdlog::info("💥 Testing floating point exception...");
            spdlog::warn("About to trigger floating point exception");
            trigger_fpe();
            break;

        default:
            spdlog::info("❌ Unknown crash type: {}", crash_type);
            return 1;
    }

    // 这行代码只有在正常执行或测试模式下才会被执行
    spdlog::info("✅ Program completed successfully!");

    // 清理资源
    SLAM_CRASH_LOGGER_SHUTDOWN();

    return 0;
}