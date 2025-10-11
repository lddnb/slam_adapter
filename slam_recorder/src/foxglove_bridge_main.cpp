#include "slam_recorder/foxglove_websocket_bridge.hpp"

#include <csignal>
#include <memory>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

using namespace ms_slam::slam_recorder;

// 全局指针用于信号处理
static std::unique_ptr<FoxgloveWebSocketBridge> g_bridge;

// 信号处理函数（SIGINT / Ctrl+C）
void signal_handler(int signal)
{
    if (signal == SIGINT) {
        spdlog::info("Caught SIGINT (Ctrl+C), shutting down...");
        if (g_bridge) {
            g_bridge->stop();
        }
    }
}

int main(int argc, char** argv)
{
    // 设置日志级别
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    spdlog::info("========================================");
    spdlog::info("Foxglove WebSocket Bridge");
    spdlog::info("========================================");

    // 注册信号处理
    std::signal(SIGINT, signal_handler);

    try {
        // 默认配置文件路径
        std::string config_file = "../config/config.yaml";

        // 加载 YAML 配置
        spdlog::info("Loading config from: {}", config_file);
        YAML::Node yaml_config = YAML::LoadFile(config_file);

        // 创建桥接配置
        FoxgloveWebSocketBridge::Config config;

        // 解析 WebSocket 配置
        if (yaml_config["websocket"]) {
            auto ws = yaml_config["websocket"];
            config.websocket.enable = ws["enable"].as<bool>(true);
            config.websocket.host = ws["host"].as<std::string>("127.0.0.1");
            config.websocket.port = ws["port"].as<uint16_t>(8765);
            config.websocket.server_name = ws["server_name"].as<std::string>("slam_recorder");
            config.websocket.poll_interval_ms = ws["poll_interval_ms"].as<uint32_t>(10);
        }

        // 解析 Recorder 配置
        if (yaml_config["recorder"]) {
            auto rec = yaml_config["recorder"];
            config.recorder.enable = rec["enable"].as<bool>(false);
            config.recorder.output_dir = rec["output_dir"].as<std::string>("./output");
            config.recorder.filename_prefix = rec["filename_prefix"].as<std::string>("recording");
            config.recorder.auto_start = rec["auto_start"].as<bool>(false);
            config.recorder.compression = rec["compression"].as<std::string>("zstd");
            config.recorder.chunk_size = rec["chunk_size"].as<uint64_t>(1048576);
        }

        // 解析 topics 配置
        if (yaml_config["topics"]) {
            for (const auto& topic_node : yaml_config["topics"]) {
                FoxgloveWebSocketBridge::TopicConfig topic;
                topic.name = topic_node["name"].as<std::string>();
                topic.schema = topic_node["schema"].as<std::string>();
                topic.enabled = topic_node["enabled"].as<bool>(true);
                config.topics.push_back(topic);
            }
        }

        // 创建桥接实例
        spdlog::info("Creating FoxgloveWebSocketBridge...");
        g_bridge = std::make_unique<FoxgloveWebSocketBridge>(config);

        // 启动桥接
        spdlog::info("Starting bridge...");
        g_bridge->start();

        spdlog::info("");
        spdlog::info("✓ Bridge is now running!");
        if (config.websocket.enable) {
            spdlog::info("  WebSocket: ws://{}:{}", config.websocket.host, config.websocket.port);
        } else {
            spdlog::info("  WebSocket: DISABLED");
        }
        if (config.recorder.enable) {
            spdlog::info("  Recorder: {}", g_bridge->is_recording() ? "RECORDING" : "Standby");
        } else {
            spdlog::info("  Recorder: DISABLED");
        }
        spdlog::info("");
        spdlog::info("Subscribed to iceoryx2 topics:");
        for (const auto& topic : config.topics) {
            if (topic.enabled) {
                spdlog::info("  - {} ({})", topic.name, topic.schema);
            }
        }
        spdlog::info("");
        spdlog::info("Press Ctrl+C to stop...");

        // 主循环：定期输出统计信息
        while (g_bridge->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto stats = g_bridge->get_statistics();
            
            uint64_t total_forwarded = stats.total_forwarded();
            uint64_t total_recorded = stats.total_recorded();
            uint64_t total_errors = stats.total_errors();

            if (total_forwarded > 0 || total_recorded > 0 || total_errors > 0) {
                spdlog::info("Statistics: WS[Fwd:{}] REC[Rec:{}] ERR[{}]",
                             config.websocket.enable ? total_forwarded : 0,
                             (config.recorder.enable && g_bridge->is_recording()) ? total_recorded : 0,
                             total_errors);
            }
        }

        spdlog::info("Bridge stopped gracefully");
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}