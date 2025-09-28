/**
 * @file crash_processor.cpp
 * @brief 独立的crash文件处理器
 *
 * 这个程序可以：
 * 1. 扫描指定目录中的slam crash文件
 * 2. 解析cpptrace堆栈信息并生成可读格式
 * 3. 将解析结果追加到spdlog日志文件中
 * 4. 处理完成后清理临时文件
 *
 * 使用方法：
 * ./crash_processor [log_file] [crash_dir] [options]
 */

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>

#include <cpptrace/cpptrace.hpp>
#include <cpptrace/formatting.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

struct ProcessorConfig {
    std::string log_file_path = "slam_crash.log";
    std::string crash_dir = "/tmp";
    bool verbose = false;
    bool keep_crash_files = false;
    std::string log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [CRASH] %v";
};

class CrashFileProcessor
{
  public:
    explicit CrashFileProcessor(const ProcessorConfig& config) : config_(config) {}

    bool initialize()
    {
        try {
            // 创建spdlog logger，用于追加到现有日志文件
            auto logger = spdlog::basic_logger_mt("crash_processor", config_.log_file_path);
            logger->set_pattern(config_.log_pattern);
            logger->set_level(spdlog::level::info);
            logger->flush_on(spdlog::level::critical);

            spdlog_logger_ = logger;

            if (config_.verbose) {
                std::cout << "Crash processor initialized" << std::endl;
                std::cout << "Log file: " << config_.log_file_path << std::endl;
                std::cout << "Crash directory: " << config_.crash_dir << std::endl;
            }

            formatter_ = cpptrace::formatter{}.addresses(cpptrace::formatter::address_mode::object).snippets(true);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize processor: " << e.what() << std::endl;
            return false;
        }
    }

    int process_crash_files()
    {
        if (!std::filesystem::exists(config_.crash_dir)) {
            if (config_.verbose) {
                std::cout << "Crash directory does not exist: " << config_.crash_dir << std::endl;
            }
            return 0;
        }

        int processed_count = 0;

        for (const auto& entry : std::filesystem::directory_iterator(config_.crash_dir)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            if (filename.find("slam_crash_") == 0 && filename.ends_with(".crash")) {
                if (process_single_crash_file(entry.path().string())) {
                    processed_count++;

                    // 处理完成后删除临时文件（除非保留选项开启）
                    if (!config_.keep_crash_files) {
                        try {
                            std::filesystem::remove(entry.path());
                            if (config_.verbose) {
                                std::cout << "Deleted crash file: " << entry.path() << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Failed to delete crash file " << entry.path() << ": " << e.what() << std::endl;
                        }
                    }
                } else {
                    std::cerr << "Failed to process crash file: " << entry.path() << std::endl;
                }
            }
        }

        if (config_.verbose) {
            std::cout << "Processed " << processed_count << " crash files" << std::endl;
        }

        return processed_count;
    }

  private:
    bool process_single_crash_file(const std::string& filepath)
    {
        try {
            if (config_.verbose) {
                std::cout << "Processing crash file: " << filepath << std::endl;
            }

            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Cannot open crash file: " << filepath << std::endl;
                return false;
            }

            // 读取文件头部信息
            std::string signal_info, pid_info, time_info, message_info;
            std::string line;
            std::streampos text_end_pos = 0;

            // 先读取文本头部
            while (std::getline(file, line)) {
                if (line.starts_with("SIGNAL:")) {
                    signal_info = line.substr(7);
                } else if (line.starts_with("PID:")) {
                    pid_info = line.substr(4);
                } else if (line.starts_with("TIME:")) {
                    time_info = line.substr(5);
                } else if (line.starts_with("MESSAGE:")) {
                    message_info = line.substr(8);
                } else if (line == "TRACE_START") {
                    text_end_pos = file.tellg();
                    break;
                }
            }

            // 尝试解析堆栈跟踪数据
            std::string trace_content = parse_trace_data(file, text_end_pos);

            // 写入spdlog日志
            write_crash_report(signal_info, pid_info, time_info, message_info, trace_content, filepath);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception processing crash file " << filepath << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::string parse_trace_data(std::ifstream& file, std::streampos text_end_pos)
    {
        std::string trace_content = "";

        if (text_end_pos <= 0) {
            return "No trace section found";
        }

        cpptrace::object_trace trace;
        while (true) {
            cpptrace::safe_object_frame frame;
            // std::size_t res = fread(&frame, sizeof(frame), 1, file);
            file.read(reinterpret_cast<char*>(&frame), sizeof(frame));
            std::size_t res = file.gcount();
            if (res == 0) {
                break;
            } else {
                trace.frames.push_back(frame.resolve());
            }
        }
        trace_content = formatter_.format(trace.resolve());

        return trace_content;
    }

    void write_crash_report(
        const std::string& signal,
        const std::string& pid,
        const std::string& time,
        const std::string& message,
        const std::string& trace,
        const std::string& filepath)
    {
        if (!spdlog_logger_) return;

        // 添加时间戳表示这是后处理的信息
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);

        spdlog_logger_->critical("=== CRASH REPORT (Post-processed) ===");
        spdlog_logger_->critical("Processed at: {}", std::ctime(&time_t_now));
        spdlog_logger_->critical("Signal: {}", signal);
        spdlog_logger_->critical("Process ID: {}", pid);
        spdlog_logger_->critical("Crash Time: {}", time);

        if (!message.empty()) {
            spdlog_logger_->critical("Message: {}", message);
        }

        spdlog_logger_->critical("Source file: {}", filepath);

        if (!trace.empty()) {
            spdlog_logger_->critical("Stack trace:\n{}", trace);
        }

        spdlog_logger_->critical("=== END CRASH REPORT ===");
        spdlog_logger_->flush();
    }

  private:
    ProcessorConfig config_;
    std::shared_ptr<spdlog::logger> spdlog_logger_;
    cpptrace::formatter formatter_;
};

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --log-file PATH    Log file path (default: slam_crash.log)\n";
    std::cout << "  --crash-dir PATH   Directory to scan for crash files (default: /tmp)\n";
    std::cout << "  --verbose          Enable verbose output\n";
    std::cout << "  --keep-files       Keep crash files after processing\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --log-file app.log --crash-dir /tmp --verbose\n";
    std::cout << "  " << program_name << " --keep-files\n";
}

int main(int argc, char* argv[])
{
    ProcessorConfig config;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--log-file" && i + 1 < argc) {
            config.log_file_path = argv[++i];
        } else if (arg == "--crash-dir" && i + 1 < argc) {
            config.crash_dir = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else if (arg == "--keep-files") {
            config.keep_crash_files = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== SLAM Crash File Processor ===" << std::endl;

    CrashFileProcessor processor(config);

    if (!processor.initialize()) {
        std::cerr << "Failed to initialize processor" << std::endl;
        return 1;
    }

    int processed = processor.process_crash_files();

    if (processed > 0) {
        std::cout << "✅ Successfully processed " << processed << " crash file(s)" << std::endl;
        std::cout << "📄 Check log file: " << config.log_file_path << std::endl;
    } else {
        std::cout << "ℹ️  No crash files found to process" << std::endl;
    }

    return 0;
}