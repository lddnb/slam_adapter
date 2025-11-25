#include "slam_recorder/foxglove_websocket_bridge.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace ms_slam::slam_recorder
{
namespace
{
/**
 * @brief 根据 schema 名称获取对应的 Protobuf 描述符
 * @param schema_name Schema 名称
 * @return 描述符指针，未知返回 nullptr
 */
const google::protobuf::Descriptor* ResolveDescriptor(const std::string& schema_name)
{
    if (schema_name == "foxglove.PointCloud") {
        return slam_common::FoxglovePointCloud::descriptor();
    }
    if (schema_name == "foxglove.CompressedImage") {
        return slam_common::FoxgloveCompressedImage::descriptor();
    }
    if (schema_name == "foxglove.Imu") {
        return slam_common::FoxgloveImu::descriptor();
    }
    if (schema_name == "foxglove.PoseInFrame") {
        return slam_common::FoxglovePoseInFrame::descriptor();
    }
    if (schema_name == "foxglove.PosesInFrame") {
        return slam_common::FoxglovePosesInFrame::descriptor();
    }
    if (schema_name == "foxglove.FrameTransforms") {
        return slam_common::FoxgloveFrameTransforms::descriptor();
    }
    if (schema_name == "foxglove.SceneUpdate") {
        return slam_common::FoxgloveSceneUpdate::descriptor();
    }
    return nullptr;
}
}  // namespace

/**
 * @brief 构造函数，完成 eCAL 初始化与 WebSocket 准备
 */
FoxgloveWebSocketBridge::FoxgloveWebSocketBridge(const Config& config) : config_(config), context_(foxglove::Context::create())
{
    spdlog::info("Initializing FoxgloveWebSocketBridge...");
    spdlog::info("  WebSocket enabled: {}", config_.websocket.enable);
    spdlog::info("  Recorder enabled: {}", config_.recorder.enable);

    const int init_code = eCAL::Initialize("foxglove_websocket_bridge");
    if (init_code < 0) {
        throw std::runtime_error("Failed to initialize eCAL");
    }
    ecal_initialized_ = true;
    owns_ecal_ = (init_code == 0);
    if (!owns_ecal_) {
        spdlog::warn("eCAL already initialized, reuse existing context");
    }

    if (config_.websocket.enable) {
        foxglove::WebSocketServerOptions options;
        options.context = context_;
        options.name = config_.websocket.server_name;
        options.host = config_.websocket.host;
        options.port = config_.websocket.port;
        options.capabilities = foxglove::WebSocketServerCapabilities::Time;
        options.supported_encodings = {"protobuf"};

        options.callbacks.onSubscribe = [](uint64_t channel_id, const foxglove::ClientMetadata& client) {
            spdlog::info("Client {} subscribed to channel {}", client.id, channel_id);
        };
        options.callbacks.onUnsubscribe = [](uint64_t channel_id, const foxglove::ClientMetadata& client) {
            spdlog::info("Client {} unsubscribed from channel {}", client.id, channel_id);
        };

        auto server_result = foxglove::WebSocketServer::create(std::move(options));
        if (!server_result.has_value()) {
            throw std::runtime_error(std::string("Failed to create WebSocket server: ") + foxglove::strerror(server_result.error()));
        }
        server_ = std::make_unique<foxglove::WebSocketServer>(std::move(server_result.value()));
        spdlog::info("✓ WebSocket server created on ws://{}:{}", config_.websocket.host, config_.websocket.port);
    } else {
        spdlog::info("WebSocket server disabled");
    }

    spdlog::info("Configuring topics...");
    for (const auto& topic : config_.topics) {
        if (!topic.enabled) {
            continue;
        }

        forwarded_count_[topic.name] = 0;
        recorded_count_[topic.name] = 0;
        error_count_[topic.name] = 0;
        pending_packets_[topic.name] = {};

        if (config_.websocket.enable) {
            const auto schema = BuildWsSchema(topic.schema);
            auto channel_result = foxglove::RawChannel::create(topic.name, "protobuf", schema, context_);
            if (!channel_result.has_value()) {
                throw std::runtime_error("Failed to create channel for topic " + topic.name + ": " + foxglove::strerror(channel_result.error()));
            }
            channels_[topic.name] = std::make_unique<foxglove::RawChannel>(std::move(channel_result.value()));
        }

        if (topic.schema == "foxglove.PointCloud") {
            pc_subs_[topic.name] = RegisterSubscriber<slam_common::FoxglovePointCloud>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.CompressedImage") {
            img_subs_[topic.name] = RegisterSubscriber<slam_common::FoxgloveCompressedImage>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.Imu") {
            imu_subs_[topic.name] = RegisterSubscriber<slam_common::FoxgloveImu>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.PoseInFrame") {
            pose_subs_[topic.name] = RegisterSubscriber<slam_common::FoxglovePoseInFrame>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.PosesInFrame") {
            poses_subs_[topic.name] = RegisterSubscriber<slam_common::FoxglovePosesInFrame>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.FrameTransforms") {
            frame_tf_subs_[topic.name] = RegisterSubscriber<slam_common::FoxgloveFrameTransforms>(topic.name, topic.schema);
        } else if (topic.schema == "foxglove.SceneUpdate") {
            frame_marker_subs_[topic.name] = RegisterSubscriber<slam_common::FoxgloveSceneUpdate>(topic.name, topic.schema);
        } else {
            spdlog::warn("Unsupported schema '{}' for topic '{}', skip", topic.schema, topic.name);
        }
    }

    spdlog::info("✓ FoxgloveWebSocketBridge initialized successfully");
}

/**
 * @brief 析构函数，确保资源释放
 */
FoxgloveWebSocketBridge::~FoxgloveWebSocketBridge()
{
    spdlog::info("Destroying FoxgloveWebSocketBridge...");
    Stop();
    if (ecal_initialized_ && owns_ecal_) {
        eCAL::Finalize();
    }
    spdlog::info("✓ FoxgloveWebSocketBridge destroyed");
}

/**
 * @brief 启动桥接器服务
 */
void FoxgloveWebSocketBridge::Start()
{
    if (running_.load()) {
        spdlog::warn("FoxgloveWebSocketBridge is already running");
        return;
    }

    running_.store(true);

    if (config_.recorder.enable && config_.recorder.auto_start) {
        StartRecording();
    }

    worker_thread_ = std::make_unique<std::thread>(&FoxgloveWebSocketBridge::Run, this);

    spdlog::info("FoxgloveWebSocketBridge started");
}

/**
 * @brief 停止桥接器服务
 */
void FoxgloveWebSocketBridge::Stop()
{
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }

    if (recording_.load()) {
        StopRecording();
    }

    if (server_) {
        auto result = server_->stop();
        if (result != foxglove::FoxgloveError::Ok) {
            spdlog::error("Error stopping WebSocket server: {}", foxglove::strerror(result));
        }
    }

    auto stats = GetStatistics();
    spdlog::info("Final Statistics:");
    for (const auto& [topic_name, topic_stats] : stats.topics) {
        spdlog::info("  - {} | forwarded:{} recorded:{} errors:{}", topic_name, topic_stats.forwarded, topic_stats.recorded, topic_stats.errors);
    }
}

/**
 * @brief 启动记录功能
 */
void FoxgloveWebSocketBridge::StartRecording()
{
    if (!config_.recorder.enable) {
        spdlog::warn("Recorder disabled in config");
        return;
    }
    if (recording_.load()) {
        spdlog::warn("Recorder already running");
        return;
    }

    try {
        InitMcapWriter();
        recording_.store(true);
        spdlog::info("Recording started: {}", current_output_file_);
    } catch (const std::exception& e) {
        spdlog::error("Failed to start recording: {}", e.what());
    }
}

/**
 * @brief 停止记录功能
 */
void FoxgloveWebSocketBridge::StopRecording()
{
    if (!recording_.load()) {
        return;
    }
    recording_.store(false);
    CloseMcapWriter();
    spdlog::info("Recording stopped: {}", current_output_file_);
}

/**
 * @brief 主循环，处理缓存并推送
 */
void FoxgloveWebSocketBridge::Run()
{
    const auto poll_interval = std::chrono::milliseconds(config_.websocket.poll_interval_ms);
    auto last_time_broadcast = std::chrono::steady_clock::now();
    const auto time_broadcast_interval = std::chrono::milliseconds(100);
    spdlog::info("Bridge main loop running, poll {} ms", config_.websocket.poll_interval_ms);

    while (running_.load()) {
        try {
            for (const auto& topic : config_.topics) {
                if (topic.enabled) {
                    PollAndForwardTopic(topic.name, topic.schema);
                }
            }

            if (config_.websocket.enable && server_) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_time_broadcast >= time_broadcast_interval) {
                    uint64_t timestamp_ns = last_global_timestamp_ns_.load(std::memory_order_relaxed);
                    if (timestamp_ns == 0U) {
                        timestamp_ns = GetCurrentTimestampNs();
                        last_global_timestamp_ns_.store(timestamp_ns, std::memory_order_relaxed);
                    }
                    server_->broadcastTime(timestamp_ns);
                    last_time_broadcast = now;
                }
            }

            std::this_thread::sleep_for(poll_interval);
        } catch (const std::exception& e) {
            spdlog::error("Error in main loop: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    spdlog::info("Bridge main loop exited");
}

/**
 * @brief 初始化 MCAP 写入器
 */
void FoxgloveWebSocketBridge::InitMcapWriter()
{
    std::filesystem::create_directories(config_.recorder.output_dir);
    current_output_file_ = GenerateOutputFilename();

    mcap_writer_ = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions opts("protobuf");
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

    topic_to_channel_id_.clear();
    mcap_schema_cache_.clear();
    next_channel_id_ = 1;
}

/**
 * @brief 关闭 MCAP 写入器
 */
void FoxgloveWebSocketBridge::CloseMcapWriter()
{
    if (mcap_writer_) {
        mcap_writer_->close();
        mcap_writer_.reset();
    }
}

/**
 * @brief 生成输出文件名
 */
std::string FoxgloveWebSocketBridge::GenerateOutputFilename() const
{
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << config_.recorder.output_dir << "/" << config_.recorder.filename_prefix << "_" << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S")
        << ".mcap";
    return oss.str();
}

/**
 * @brief 将数据写入 MCAP 文件
 */
void FoxgloveWebSocketBridge::RecordToMcap(
    const std::string& topic_name,
    const std::string& schema_name,
    const std::string& data,
    uint64_t timestamp_ns)
{
    if (!mcap_writer_) {
        return;
    }

    mcap::SchemaId schema_id;
    auto schema_it = mcap_schema_cache_.find(schema_name);
    if (schema_it == mcap_schema_cache_.end()) {
        const auto* descriptor = ResolveDescriptor(schema_name);
        if (descriptor == nullptr) {
            spdlog::warn("Skip recording for unknown schema {}", schema_name);
            return;
        }
        const std::string descriptor_set = BuildDescriptorSet(descriptor);
        mcap::Schema schema(schema_name, "protobuf", descriptor_set);
        mcap_writer_->addSchema(schema);
        schema_id = schema.id;
        mcap_schema_cache_[schema_name] = schema_id;
    } else {
        schema_id = schema_it->second;
    }

    uint16_t channel_id;
    auto it = topic_to_channel_id_.find(topic_name);
    if (it == topic_to_channel_id_.end()) {
        channel_id = next_channel_id_++;
        mcap::Channel channel(topic_name, "protobuf", schema_id, {});
        mcap_writer_->addChannel(channel);
        topic_to_channel_id_[topic_name] = channel_id;
    } else {
        channel_id = it->second;
    }

    mcap::Message msg;
    msg.channelId = channel_id;
    msg.sequence = recorded_count_.at(topic_name).load();
    msg.logTime = timestamp_ns;
    msg.publishTime = timestamp_ns;
    msg.data = reinterpret_cast<const std::byte*>(data.data());
    msg.dataSize = data.size();

    auto status = mcap_writer_->write(msg);
    if (status.ok()) {
        recorded_count_.at(topic_name)++;
    } else {
        error_count_.at(topic_name)++;
        spdlog::error("Failed to write message for topic '{}' to MCAP: {}", topic_name, status.message);
    }
}

uint64_t FoxgloveWebSocketBridge::GetCurrentTimestampNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

uint64_t FoxgloveWebSocketBridge::ExtractTimestampNs(const std::string& schema, const google::protobuf::Message& message) const
{
    if (schema == "foxglove.PointCloud") {
        const auto* msg = dynamic_cast<const slam_common::FoxglovePointCloud*>(&message);
        if (msg != nullptr) {
            return ToNanoseconds(msg->timestamp());
        }
    } else if (schema == "foxglove.CompressedImage") {
        const auto* msg = dynamic_cast<const slam_common::FoxgloveCompressedImage*>(&message);
        if (msg != nullptr) {
            return ToNanoseconds(msg->timestamp());
        }
    } else if (schema == "foxglove.Imu") {
        const auto* msg = dynamic_cast<const slam_common::FoxgloveImu*>(&message);
        if (msg != nullptr) {
            return ToNanoseconds(msg->timestamp());
        }
    } else if (schema == "foxglove.PoseInFrame") {
        const auto* msg = dynamic_cast<const slam_common::FoxglovePoseInFrame*>(&message);
        if (msg != nullptr) {
            return ToNanoseconds(msg->timestamp());
        }
    } else if (schema == "foxglove.PosesInFrame") {
        const auto* msg = dynamic_cast<const slam_common::FoxglovePosesInFrame*>(&message);
        if (msg != nullptr) {
            return ToNanoseconds(msg->timestamp());
        }
    } else if (schema == "foxglove.FrameTransforms") {
        const auto* msg = dynamic_cast<const slam_common::FoxgloveFrameTransforms*>(&message);
        if (msg != nullptr && msg->transforms_size() > 0) {
            return ToNanoseconds(msg->transforms(0).timestamp());
        }
    } else if (schema == "foxglove.SceneUpdate") {
        const auto* msg = dynamic_cast<const slam_common::FoxgloveSceneUpdate*>(&message);
        if (msg != nullptr && msg->entities_size() > 0) {
            return ToNanoseconds(msg->entities(0).timestamp());
        }
    }
    return 0U;
}

uint64_t FoxgloveWebSocketBridge::AlignTimestamp(uint64_t message_time_ns)
{
    if (message_time_ns == 0U) {
        return GetCurrentTimestampNs();
    }

    if (!time_sync_initialized_.load(std::memory_order_acquire)) {
        const uint64_t now = GetCurrentTimestampNs();
        const uint64_t offset = (now > message_time_ns) ? now - message_time_ns : 0U;
        time_offset_ns_.store(offset, std::memory_order_release);
        time_sync_initialized_.store(true, std::memory_order_release);
        spdlog::info("Initialized timestamp offset: now={} message={} offset={}", now, message_time_ns, offset);
    }

    return message_time_ns + time_offset_ns_.load(std::memory_order_acquire);
}

uint64_t FoxgloveWebSocketBridge::EnsureGlobalMonotonic(uint64_t timestamp_ns)
{
    uint64_t current = last_global_timestamp_ns_.load(std::memory_order_relaxed);
    while (timestamp_ns <= current) {
        timestamp_ns = current + 1U;
    }
    last_global_timestamp_ns_.store(timestamp_ns, std::memory_order_relaxed);
    return timestamp_ns;
}

foxglove::Schema FoxgloveWebSocketBridge::BuildWsSchema(const std::string& schema_name)
{
    auto& buffer = schema_descriptor_cache_[schema_name];
    if (buffer.empty()) {
        const auto* descriptor = ResolveDescriptor(schema_name);
        if (descriptor == nullptr) {
            throw std::runtime_error("Unknown schema for WebSocket: " + schema_name);
        }
        buffer = BuildDescriptorSet(descriptor);
    }

    foxglove::Schema schema;
    schema.name = schema_name;
    schema.encoding = "protobuf";
    schema.data = reinterpret_cast<const std::byte*>(buffer.data());
    schema.data_len = buffer.size();
    return schema;
}

std::string FoxgloveWebSocketBridge::BuildDescriptorSet(const google::protobuf::Descriptor* descriptor)
{
    if (descriptor == nullptr) {
        return {};
    }
    google::protobuf::FileDescriptorSet descriptor_set;
    std::queue<const google::protobuf::FileDescriptor*> pending;
    std::unordered_set<std::string> visited;
    pending.push(descriptor->file());
    visited.insert(descriptor->file()->name());

    while (!pending.empty()) {
        const auto* file = pending.front();
        pending.pop();
        file->CopyTo(descriptor_set.add_file());
        for (int i = 0; i < file->dependency_count(); ++i) {
            const auto* dependency = file->dependency(i);
            if (visited.insert(dependency->name()).second) {
                pending.push(dependency);
            }
        }
    }

    return descriptor_set.SerializeAsString();
}

uint64_t FoxgloveWebSocketBridge::ToNanoseconds(const google::protobuf::Timestamp& stamp)
{
    if (stamp.seconds() < 0 || stamp.nanos() < 0) {
        return 0U;
    }
    constexpr uint64_t kNsPerSec = 1'000'000'000ULL;
    return static_cast<uint64_t>(stamp.seconds()) * kNsPerSec + static_cast<uint64_t>(stamp.nanos());
}

bool FoxgloveWebSocketBridge::SerializeMessage(const google::protobuf::Message& message, std::string& buffer)
{
    buffer.clear();
    if (!message.SerializeToString(&buffer)) {
        spdlog::error("Failed to serialize message {}", message.GetTypeName());
        return false;
    }
    return true;
}

void FoxgloveWebSocketBridge::PollAndForwardTopic(const std::string& topic_name, const std::string& schema)
{
    std::vector<PendingPacket> packets;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto it = pending_packets_.find(topic_name);
        if (it != pending_packets_.end()) {
            packets.swap(it->second);
        }
    }

    if (packets.empty()) {
        return;
    }

    const bool has_websocket_sinks = config_.websocket.enable && channels_.count(topic_name) && channels_.at(topic_name)->has_sinks();

    for (const auto& packet : packets) {
        if (has_websocket_sinks) {
            auto result =
                channels_.at(topic_name)->log(reinterpret_cast<const std::byte*>(packet.data.data()), packet.data.size(), packet.timestamp_ns);
            if (result == foxglove::FoxgloveError::Ok) {
                forwarded_count_.at(topic_name)++;
            } else {
                error_count_.at(topic_name)++;
                spdlog::error("Failed to forward {} on '{}': {}", schema, topic_name, foxglove::strerror(result));
            }
        }

        if (recording_.load()) {
            RecordToMcap(topic_name, schema, packet.data, packet.timestamp_ns);
        }

        last_message_time_ns_.store(packet.timestamp_ns, std::memory_order_relaxed);
    }
}

FoxgloveWebSocketBridge::Statistics FoxgloveWebSocketBridge::GetStatistics() const
{
    Statistics stats;
    for (const auto& topic : config_.topics) {
        if (topic.enabled) {
            stats.topics[topic.name] = {
                .forwarded = forwarded_count_.at(topic.name).load(),
                .recorded = recorded_count_.at(topic.name).load(),
                .errors = error_count_.at(topic.name).load()};
        }
    }
    return stats;
}

/**
 * @brief 模板化订阅注册
 */
template <typename MessageType>
std::shared_ptr<eCAL::protobuf::CSubscriber<MessageType>> FoxgloveWebSocketBridge::RegisterSubscriber(
    const std::string& topic_name,
    const std::string& schema_name)
{
    auto subscriber = std::make_shared<eCAL::protobuf::CSubscriber<MessageType>>(topic_name);
    subscriber->SetReceiveCallback([this, topic_name, schema_name](const eCAL::STopicId&, const MessageType& msg, long long send_time, long long) {
        const uint64_t message_ts = ExtractTimestampNs(schema_name, msg);
        const uint64_t fallback_ts = (send_time > 0) ? static_cast<uint64_t>(send_time) * 1000ULL : GetCurrentTimestampNs();
        const uint64_t aligned_ts = EnsureGlobalMonotonic(AlignTimestamp(message_ts == 0U ? fallback_ts : message_ts));

        std::string buffer;
        if (!SerializeMessage(msg, buffer)) {
            error_count_.at(topic_name)++;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_packets_[topic_name].push_back({std::move(buffer), aligned_ts});
        }
    });

    spdlog::info("eCAL subscriber created for topic {} ({})", topic_name, schema_name);
    return subscriber;
}

// 显式实例化
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxglovePointCloud>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxglovePointCloud>(const std::string&, const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxgloveCompressedImage>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxgloveCompressedImage>(const std::string&, const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxgloveImu>> FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxgloveImu>(
    const std::string&,
    const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxglovePoseInFrame>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxglovePoseInFrame>(const std::string&, const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxglovePosesInFrame>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxglovePosesInFrame>(const std::string&, const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxgloveFrameTransforms>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxgloveFrameTransforms>(const std::string&, const std::string&);
template std::shared_ptr<eCAL::protobuf::CSubscriber<slam_common::FoxgloveSceneUpdate>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FoxgloveSceneUpdate>(const std::string&, const std::string&);

}  // namespace ms_slam::slam_recorder
