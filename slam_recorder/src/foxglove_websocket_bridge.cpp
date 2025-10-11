#include "slam_recorder/foxglove_websocket_bridge.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#define MCAP_IMPLEMENTATION
#include <mcap/writer.hpp>

namespace ms_slam::slam_recorder
{

// ============================================================================
// 构造函数
// ============================================================================

FoxgloveWebSocketBridge::FoxgloveWebSocketBridge(const Config& config)
: config_(config), context_(foxglove::Context::create())
{
    spdlog::info("Initializing FoxgloveWebSocketBridge...");
    spdlog::info("  WebSocket enabled: {}", config_.websocket.enable);
    spdlog::info("  Recorder enabled: {}", config_.recorder.enable);

    // -------------------------------------------------------------------------
    // 1. 创建 iceoryx2 node
    // -------------------------------------------------------------------------
    try {
        node_ = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
            iox2::NodeBuilder()
                .create<iox2::ServiceType::Ipc>()
                .expect("Failed to create iceoryx2 node")
        );
        spdlog::info("✓ iceoryx2 node created");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create iceoryx2 node: ") + e.what());
    }

    // -------------------------------------------------------------------------
    // 2. 创建 WebSocket 服务器（如果启用）
    // -------------------------------------------------------------------------
    if (config_.websocket.enable) {
        foxglove::WebSocketServerOptions options;
        options.context = context_;
        options.name = config_.websocket.server_name;
        options.host = config_.websocket.host;
        options.port = config_.websocket.port;
        options.capabilities = foxglove::WebSocketServerCapabilities::Time;
        options.supported_encodings = {"flatbuffer"};

        options.callbacks.onSubscribe = [](uint64_t channel_id, const foxglove::ClientMetadata& client) {
            spdlog::info("Client {} subscribed to channel {}", client.id, channel_id);
        };
        options.callbacks.onUnsubscribe = [](uint64_t channel_id, const foxglove::ClientMetadata& client) {
            spdlog::info("Client {} unsubscribed from channel {}", client.id, channel_id);
        };

        auto server_result = foxglove::WebSocketServer::create(std::move(options));
        if (!server_result.has_value()) {
            throw std::runtime_error(std::string("Failed to create WebSocket server: ") +
                                     foxglove::strerror(server_result.error()));
        }
        server_ = std::make_unique<foxglove::WebSocketServer>(std::move(server_result.value()));
        spdlog::info("✓ WebSocket server created on ws://{}:{}", config_.websocket.host, config_.websocket.port);
    } else {
        spdlog::info("WebSocket server disabled");
    }

    // -------------------------------------------------------------------------
    // 3. 根据配置初始化 Topics
    // -------------------------------------------------------------------------
    spdlog::info("Configuring topics...");
    for (const auto& topic : config_.topics) {
        if (!topic.enabled) {
            continue;
        }

        spdlog::info("  - Enabling topic: {} ({})", topic.name, topic.schema);

        // 初始化统计计数器
        forwarded_count_[topic.name] = 0;
        recorded_count_[topic.name] = 0;
        error_count_[topic.name] = 0;

        foxglove::Schema schema;
        schema.encoding = "flatbuffer";

        // 根据 schema 类型创建 channel 和 subscriber
        if (topic.schema == "foxglove.PointCloud") {
            schema.name = "foxglove.PointCloud";
            schema.data = reinterpret_cast<const std::byte*>(foxglove::PointCloudBinarySchema::data());
            schema.data_len = foxglove::PointCloudBinarySchema::size();
            pc_subs_[topic.name] = std::make_unique<slam_common::FBSSubscriber<slam_common::FoxglovePointCloud>>(node_, topic.name);

        } else if (topic.schema == "foxglove.CompressedImage") {
            schema.name = "foxglove.CompressedImage";
            schema.data = reinterpret_cast<const std::byte*>(foxglove::CompressedImageBinarySchema::data());
            schema.data_len = foxglove::CompressedImageBinarySchema::size();
            img_subs_[topic.name] = std::make_unique<slam_common::FBSSubscriber<slam_common::FoxgloveCompressedImage>>(node_, topic.name);

        } else if (topic.schema == "foxglove.Imu") {
            schema.name = "foxglove.Imu";
            schema.data = reinterpret_cast<const std::byte*>(foxglove::ImuBinarySchema::data());
            schema.data_len = foxglove::ImuBinarySchema::size();
            imu_subs_[topic.name] = std::make_unique<slam_common::FBSSubscriber<slam_common::FoxgloveImu>>(node_, topic.name);

        } else {
            spdlog::warn("  - Unsupported schema type '{}' for topic '{}'", topic.schema, topic.name);
            continue;
        }

        // 如果 WebSocket 启用，则创建 channel
        if (config_.websocket.enable) {
            auto channel_result = foxglove::RawChannel::create(topic.name, "flatbuffer", std::move(schema), context_);
            if (!channel_result.has_value()) {
                throw std::runtime_error("Failed to create channel for topic " + topic.name + ": " +
                                         foxglove::strerror(channel_result.error()));
            }
            channels_[topic.name] = std::make_unique<foxglove::RawChannel>(std::move(channel_result.value()));
            spdlog::info("    ✓ WebSocket channel created");
        }
    }

    spdlog::info("✓ FoxgloveWebSocketBridge initialized successfully");
}


// ============================================================================
// 析构函数
// ============================================================================

FoxgloveWebSocketBridge::~FoxgloveWebSocketBridge()
{
    spdlog::info("Destroying FoxgloveWebSocketBridge...");
    stop();
    spdlog::info("✓ FoxgloveWebSocketBridge destroyed");
}

// ============================================================================
// 启动/停止方法
// ============================================================================

void FoxgloveWebSocketBridge::start()
{
    if (running_.load()) {
        spdlog::warn("FoxgloveWebSocketBridge is already running");
        return;
    }

    spdlog::info("Starting FoxgloveWebSocketBridge...");
    running_.store(true);

    // 如果启用录制且设置为自动开始，则初始化 MCAP writer
    if (config_.recorder.enable && config_.recorder.auto_start) {
        start_recording();
    }

    // 在独立线程中运行主循环
    worker_thread_ = std::make_unique<std::thread>(&FoxgloveWebSocketBridge::run, this);

    spdlog::info("✓ FoxgloveWebSocketBridge started");
    if (config_.websocket.enable) {
        spdlog::info("  WebSocket: ws://{}:{}", config_.websocket.host, config_.websocket.port);
    }
    if (config_.recorder.enable) {
        spdlog::info("  Recorder: {}", recording_.load() ? "RECORDING" : "Standby");
    }
}

void FoxgloveWebSocketBridge::stop()
{
    if (!running_.load()) {
        return;
    }

    spdlog::info("Stopping FoxgloveWebSocketBridge...");
    running_.store(false);

    // 停止录制
    if (recording_.load()) {
        stop_recording();
    }

    // 等待工作线程退出
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }

    // 停止 WebSocket 服务器
    if (server_) {
        auto result = server_->stop();
        if (result != foxglove::FoxgloveError::Ok) {
            spdlog::error("Error stopping WebSocket server: {}", foxglove::strerror(result));
        }
    }

    spdlog::info("✓ FoxgloveWebSocketBridge stopped");

    // 输出统计信息
    auto stats = get_statistics();
    spdlog::info("Final Statistics:");
    for (const auto& [topic_name, topic_stats] : stats.topics) {
        spdlog::info("  - Topic [{}]:", topic_name);
        if (config_.websocket.enable) {
            spdlog::info("    Forwarded: {}, Errors: {}", topic_stats.forwarded, topic_stats.errors);
        }
        if (config_.recorder.enable) {
            spdlog::info("    Recorded: {}", topic_stats.recorded);
        }
    }
}

// ============================================================================
// 录制控制方法
// ============================================================================

void FoxgloveWebSocketBridge::start_recording()
{
    if (!config_.recorder.enable) {
        spdlog::warn("Recorder is disabled in config");
        return;
    }

    if (recording_.load()) {
        spdlog::warn("Already recording");
        return;
    }

    try {
        init_mcap_writer();
        recording_.store(true);
        spdlog::info("✓ Recording started: {}", current_output_file_);
    } catch (const std::exception& e) {
        spdlog::error("Failed to start recording: {}", e.what());
    }
}

void FoxgloveWebSocketBridge::stop_recording()
{
    if (!recording_.load()) {
        return;
    }

    spdlog::info("Stopping recording...");
    recording_.store(false);

    close_mcap_writer();

    spdlog::info("✓ Recording stopped: {}", current_output_file_);
}

// ============================================================================
// 主运行循环
// ============================================================================

void FoxgloveWebSocketBridge::run()
{
    spdlog::info("Main loop started (poll interval: {} ms)", config_.websocket.poll_interval_ms);

    const auto poll_interval = std::chrono::milliseconds(config_.websocket.poll_interval_ms);
    auto last_time_broadcast = std::chrono::steady_clock::now();
    const auto time_broadcast_interval = std::chrono::milliseconds(100);

    while (running_.load()) {
        try {
            // 轮询并处理所有 topic
            for (const auto& topic : config_.topics) {
                if (topic.enabled) {
                    poll_and_forward_topic(topic.name, topic.schema);
                }
            }

            // 定期广播服务器时间（如果 WebSocket 启用）
            if (config_.websocket.enable && server_) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_time_broadcast >= time_broadcast_interval) {
                    uint64_t timestamp_ns = get_current_timestamp_ns();
                    server_->broadcastTime(timestamp_ns);
                    last_time_broadcast = now;
                }
            }

            // 休眠以避免 CPU 空转
            std::this_thread::sleep_for(poll_interval);
        } catch (const std::exception& e) {
            spdlog::error("Error in main loop: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    spdlog::info("Main loop exited");
}

// ============================================================================
// 轮询并转发方法
// ============================================================================

void FoxgloveWebSocketBridge::poll_and_forward_topic(const std::string& topic_name, const std::string& schema)
{
    // 检查是否有 WebSocket 客户端订阅或是否正在录制
    bool has_websocket_sinks = config_.websocket.enable && channels_.count(topic_name) &&
                               channels_.at(topic_name)->has_sinks();
    if (!has_websocket_sinks && !recording_.load()) {
        return;
    }

    try {
        // 根据 schema 类型选择正确的订阅者并处理数据
        if (schema == "foxglove.PointCloud" && pc_subs_.count(topic_name)) {
            auto samples = pc_subs_.at(topic_name)->receive_all();
            for (auto& sample : samples) {
                auto buffer = sample.serialize();
                uint64_t timestamp_ns = get_current_timestamp_ns();

                if (has_websocket_sinks) {
                    auto result = channels_.at(topic_name)->log(
                        reinterpret_cast<const std::byte*>(buffer.data()),
                        buffer.size(),
                        timestamp_ns
                    );
                    if (result == foxglove::FoxgloveError::Ok) {
                        forwarded_count_.at(topic_name)++;
                    } else {
                        error_count_.at(topic_name)++;
                        spdlog::error("Failed to forward PointCloud on '{}': {}",
                                      topic_name, foxglove::strerror(result));
                    }
                }

                if (recording_.load()) {
                    record_to_mcap(topic_name, schema, buffer.data(), buffer.size(), timestamp_ns);
                }
            }
        } else if (schema == "foxglove.CompressedImage" && img_subs_.count(topic_name)) {
            auto samples = img_subs_.at(topic_name)->receive_all();
            for (auto& sample : samples) {
                auto buffer = sample.serialize();
                uint64_t timestamp_ns = get_current_timestamp_ns();

                if (has_websocket_sinks) {
                    auto result = channels_.at(topic_name)->log(
                        reinterpret_cast<const std::byte*>(buffer.data()),
                        buffer.size(),
                        timestamp_ns
                    );
                    if (result == foxglove::FoxgloveError::Ok) {
                        forwarded_count_.at(topic_name)++;
                    } else {
                        error_count_.at(topic_name)++;
                        spdlog::error("Failed to forward CompressedImage on '{}': {}",
                                      topic_name, foxglove::strerror(result));
                    }
                }

                if (recording_.load()) {
                    record_to_mcap(topic_name, schema, buffer.data(), buffer.size(), timestamp_ns);
                }
            }
        } else if (schema == "foxglove.Imu" && imu_subs_.count(topic_name)) {
            auto samples = imu_subs_.at(topic_name)->receive_all();
            for (auto& sample : samples) {
                auto buffer = sample.serialize();
                uint64_t timestamp_ns = get_current_timestamp_ns();

                if (has_websocket_sinks) {
                    auto result = channels_.at(topic_name)->log(
                        reinterpret_cast<const std::byte*>(buffer.data()),
                        buffer.size(),
                        timestamp_ns
                    );
                    if (result == foxglove::FoxgloveError::Ok) {
                        forwarded_count_.at(topic_name)++;
                    } else {
                        error_count_.at(topic_name)++;
                        spdlog::error("Failed to forward IMU on '{}': {}",
                                      topic_name, foxglove::strerror(result));
                    }
                }

                if (recording_.load()) {
                    record_to_mcap(topic_name, schema, buffer.data(), buffer.size(), timestamp_ns);
                }
            }
        }
    } catch (const std::exception& e) {
        error_count_.at(topic_name)++;
        spdlog::error("Error polling topic '{}': {}", topic_name, e.what());
    }
}


// ============================================================================
// MCAP 录制方法
// ============================================================================

void FoxgloveWebSocketBridge::init_mcap_writer()
{
    // 创建输出目录
    std::filesystem::create_directories(config_.recorder.output_dir);

    // 生成文件名
    current_output_file_ = generate_output_filename();

    // 创建 MCAP writer
    mcap_writer_ = std::make_unique<mcap::McapWriter>();

    // 设置压缩
    mcap::McapWriterOptions opts("flatbuffers");
    if (config_.recorder.compression == "zstd") {
        opts.compression = mcap::Compression::Zstd;
    } else if (config_.recorder.compression == "lz4") {
        opts.compression = mcap::Compression::Lz4;
    } else {
        opts.compression = mcap::Compression::None;
    }
    opts.chunkSize = config_.recorder.chunk_size;

    auto status = mcap_writer_->open(current_output_file_, opts);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create MCAP file: " + status.message);
    }

    // 重置计数器
    topic_to_channel_id_.clear();
    next_channel_id_ = 1;

    spdlog::info("MCAP writer initialized");
    spdlog::info("  Output file: {}", current_output_file_);
    spdlog::info("  Compression: {}", config_.recorder.compression);
}

void FoxgloveWebSocketBridge::close_mcap_writer()
{
    if (mcap_writer_) {
        mcap_writer_->close();
        mcap_writer_.reset();
    }
}

std::string FoxgloveWebSocketBridge::generate_output_filename() const
{
    // 生成时间戳：recording_20251011_143020.mcap
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << config_.recorder.output_dir << "/"
        << config_.recorder.filename_prefix << "_"
        << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S")
        << ".mcap";

    return oss.str();
}

void FoxgloveWebSocketBridge::record_to_mcap(const std::string& topic_name, const std::string& schema_name,
                                           const uint8_t* data, size_t size, uint64_t timestamp_ns)
{
    if (!mcap_writer_) {
        return;
    }

    // 创建/获取 channel
    uint16_t channel_id;
    auto it = topic_to_channel_id_.find(topic_name);
    if (it == topic_to_channel_id_.end()) {
        channel_id = next_channel_id_++;
        topic_to_channel_id_[topic_name] = channel_id;

        // 根据 schema 名称添加 schema
        mcap::Schema mcap_schema;
        if (schema_name == "foxglove.PointCloud") {
            foxglove::PointCloudBinarySchema schema_data;
            mcap_schema = mcap::Schema(schema_name, "flatbuffer",
                std::string(reinterpret_cast<const char*>(schema_data.data()), schema_data.size()));
        } else if (schema_name == "foxglove.CompressedImage") {
            foxglove::CompressedImageBinarySchema schema_data;
            mcap_schema = mcap::Schema(schema_name, "flatbuffer",
                std::string(reinterpret_cast<const char*>(schema_data.data()), schema_data.size()));
        } else if (schema_name == "foxglove.Imu") {
            foxglove::ImuBinarySchema schema_data;
            mcap_schema = mcap::Schema(schema_name, "flatbuffer",
                std::string(reinterpret_cast<const char*>(schema_data.data()), schema_data.size()));
        } else {
            // 不支持的 schema，不录制
            return;
        }
        mcap_writer_->addSchema(mcap_schema);

        // 添加 channel
        mcap::Channel channel(topic_name, "flatbuffer", mcap_schema.id, {});
        mcap_writer_->addChannel(channel);
    } else {
        channel_id = it->second;
    }

    // 写入消息
    mcap::Message msg;
    msg.channelId = channel_id;
    msg.sequence = recorded_count_.at(topic_name).load();
    msg.logTime = timestamp_ns;
    msg.publishTime = timestamp_ns;
    msg.data = reinterpret_cast<const std::byte*>(data);
    msg.dataSize = size;

    auto status = mcap_writer_->write(msg);
    if (status.ok()) {
        recorded_count_.at(topic_name)++;
    } else {
        error_count_.at(topic_name)++;
        spdlog::error("Failed to write message for topic '{}' to MCAP: {}", topic_name, status.message);
    }
}


// ============================================================================
// 辅助方法
// ============================================================================

uint64_t FoxgloveWebSocketBridge::get_current_timestamp_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

FoxgloveWebSocketBridge::Statistics FoxgloveWebSocketBridge::get_statistics() const
{
    Statistics stats;
    for (const auto& topic : config_.topics) {
        if (topic.enabled) {
            stats.topics[topic.name] = {
                .forwarded = forwarded_count_.at(topic.name).load(),
                .recorded = recorded_count_.at(topic.name).load(),
                .errors = error_count_.at(topic.name).load()
            };
        }
    }
    return stats;
}

}  // namespace ms_slam::slam_recorder