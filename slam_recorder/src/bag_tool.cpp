#include "slam_recorder/bag_tool.hpp"

#include <poll.h>
#include <csignal>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/message.h>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <slam_common/foxglove_messages.hpp>

#include "slam_recorder/ros1_msg.hpp"

namespace ms_slam::slam_recorder
{

namespace
{

using ms_slam::slam_common::FoxgloveCompressedImage;
using ms_slam::slam_common::FoxgloveFrameTransforms;
using ms_slam::slam_common::FoxgloveImu;
using ms_slam::slam_common::FoxglovePointCloud;
using ms_slam::slam_common::FoxglovePoseInFrame;
using ms_slam::slam_common::FoxglovePosesInFrame;

/**
 * @brief 内部使用的消息类型分类
 */
enum class MessageKind { PointCloud, CompressedImage, Imu, PoseInFrame, PosesInFrame, FrameTransforms, Unsupported };

/**
 * @brief 消息描述信息，用于快速判断数据来源类型
 */
struct MessageDescriptor {
    MessageKind kind{MessageKind::Unsupported};
    bool is_ros{false};
    bool is_protobuf{false};
    bool is_livox{false};
};

/**
 * @brief 将输入字符串转换为小写，用于配置解析
 * @param value 待转换的原始字符串
 * @return 转换为小写后的字符串
 */
std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

/**
 * @brief 将配置字符串解析为 InputType 枚举
 * @param value 配置中的类型字符串
 * @return 对应的 InputType 值
 * @note 不支持的字符串会抛出 std::runtime_error 异常
 */
InputType ParseInputType(const std::string& value)
{
    const auto lowered = ToLower(value);
    if (lowered == "ros1_mcap" || lowered == "ros1-mcap") {
        return InputType::Ros1Mcap;
    }
    if (lowered == "protobuf_mcap" || lowered == "protobuf" || lowered == "foxglove_mcap") {
        return InputType::ProtobufMcap;
    }
    if (lowered == "flatbuffer_mcap" || lowered == "flatbuffers_mcap" || lowered == "foxglove_flatbuffer_mcap") {
        spdlog::warn("Deprecated input type flatbuffer_mcap, fallback to protobuf");
        return InputType::ProtobufMcap;
    }
    if (lowered == "rosbag" || lowered == "ros1_bag") {
        return InputType::Rosbag;
    }
    throw std::runtime_error("unsupported input.type value: " + value);
}

/**
 * @brief 根据 MCAP 消息元数据识别消息类型
 * @param view MCAP 消息视图
 * @return 描述消息特征的 MessageDescriptor
 */
MessageDescriptor DescribeMessage(const mcap::MessageView& view)
{
    MessageDescriptor descriptor;
    const std::string schema_name = view.schema ? view.schema->name : "";
    const std::string encoding = view.channel ? view.channel->messageEncoding : "";

    auto set_ros = [&](MessageKind kind) {
        descriptor.kind = kind;
        descriptor.is_ros = true;
    };

    if (schema_name == "sensor_msgs/PointCloud2") {
        set_ros(MessageKind::PointCloud);
        return descriptor;
    }
    if (schema_name == "sensor_msgs/CompressedImage") {
        set_ros(MessageKind::CompressedImage);
        return descriptor;
    }
    if (schema_name == "sensor_msgs/Imu") {
        set_ros(MessageKind::Imu);
        return descriptor;
    }
    if (schema_name == "nav_msgs/Path") {
        set_ros(MessageKind::PosesInFrame);
        return descriptor;
    }
    if (schema_name == "nav_msgs/Odometry") {
        set_ros(MessageKind::PoseInFrame);
        return descriptor;
    }
    if (schema_name == "livox_ros_driver2/CustomMsg" || schema_name == "livox_ros_driver/CustomMsg") {
        set_ros(MessageKind::PointCloud);
        descriptor.is_livox = true;
        return descriptor;
    }
    if (schema_name == "tf2_msgs/TFMessage") {
        set_ros(MessageKind::FrameTransforms);
        return descriptor;
    }

    if (encoding == "protobuf") {
        if (schema_name == "foxglove.PointCloud") {
            descriptor.kind = MessageKind::PointCloud;
            descriptor.is_protobuf = true;
            return descriptor;
        }
        if (schema_name == "foxglove.CompressedImage") {
            descriptor.kind = MessageKind::CompressedImage;
            descriptor.is_protobuf = true;
            return descriptor;
        }
        if (schema_name == "foxglove.Imu") {
            descriptor.kind = MessageKind::Imu;
            descriptor.is_protobuf = true;
            return descriptor;
        }
        if (schema_name == "foxglove.PoseInFrame") {
            descriptor.kind = MessageKind::PoseInFrame;
            descriptor.is_protobuf = true;
            return descriptor;
        }
        if (schema_name == "foxglove.PosesInFrame") {
            descriptor.kind = MessageKind::PosesInFrame;
            descriptor.is_protobuf = true;
            return descriptor;
        }
        if (schema_name == "foxglove.FrameTransforms") {
            descriptor.kind = MessageKind::FrameTransforms;
            descriptor.is_protobuf = true;
            return descriptor;
        }
    }

    descriptor.kind = MessageKind::Unsupported;
    return descriptor;
}

/**
 * @brief 基于全局配置解析单个话题的实际配置
 * @param config 工具全局配置
 * @param topic 话题名称
 * @return 合并后的 TopicSettings
 */
TopicSettings ResolveTopic(const ToolConfig& config, const std::string& topic)
{
    TopicSettings resolved;
    resolved.playback = config.playback_enabled && config.default_playback;
    resolved.record = config.record_enabled && config.default_record;
    resolved.publish_service = topic;

    const auto it = config.topics.find(topic);
    if (it != config.topics.end()) {
        if (!it->second.publish_service.empty()) {
            resolved.publish_service = it->second.publish_service;
        }
        resolved.playback = config.playback_enabled && it->second.playback;
        resolved.record = config.record_enabled && it->second.record;
    }

    return resolved;
}

/**
 * @brief 缓存 MCAP Schema ID，减少重复注册
 */
struct SchemaCache {
    std::optional<mcap::SchemaId> pointcloud;
    std::optional<mcap::SchemaId> compressed_image;
    std::optional<mcap::SchemaId> imu;
    std::optional<mcap::SchemaId> pose_in_frame;
    std::optional<mcap::SchemaId> poses_in_frame;
    std::optional<mcap::SchemaId> frame_transforms;
};

/**
 * @brief 构造包含依赖的 Protobuf 描述符集
 * @param descriptor 顶层描述符
 * @return 序列化后的描述符集
 */
std::string BuildDescriptorSetString(const google::protobuf::Descriptor* descriptor)
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

/**
 * @brief 录制上下文信息，封装 writer 与 channel 管理
 */
struct RecorderContext {
    bool enabled{false};
    std::unique_ptr<mcap::McapWriter> writer;
    SchemaCache schemas;
    std::unordered_map<std::string, uint16_t> channel_ids;
    std::unordered_map<std::string, uint32_t> sequence_numbers;
    uint16_t next_channel_id{1};
};

/**
 * @brief 回放发布器基类，提供统一的 Protobuf 发布接口
 */
struct PlaybackPublisherBase {
    virtual ~PlaybackPublisherBase() = default;
    virtual bool Publish(const google::protobuf::Message& message) = 0;
};

/**
 * @brief 模板化的回放发布器，封装 eCAL Protobuf 发布
 * @tparam MessageType Protobuf 消息类型
 */
template <typename MessageType>
class PlaybackPublisher final : public PlaybackPublisherBase
{
  public:
    /**
     * @brief 构造函数
     * @param service_name 服务名称
     */
    explicit PlaybackPublisher(const std::string& service_name) : publisher_(service_name) {}

    /**
     * @brief 发布 Protobuf 消息
     * @param message 待发布的消息
     * @return 发布成功返回 true
     */
    bool Publish(const google::protobuf::Message& message) override
    {
        const auto* typed_message = dynamic_cast<const MessageType*>(&message);
        if (typed_message == nullptr) {
            spdlog::error("Publish failed: message type mismatch");
            return false;
        }
        if (publisher_.GetSubscriberCount() == 0) {
            // 无订阅者时跳过发送，避免误报失败
            return true;
        }
        return publisher_.Send(*typed_message);
    }

  private:
    eCAL::protobuf::CPublisher<MessageType> publisher_;
};

/**
 * @brief 回放上下文，包含调度、节点及发布器缓存
 */
struct PlaybackContext {
    bool enabled{false};
    bool sync_time{true};
    double rate{1.0};
    bool ecal_initialized{false};
    bool owns_ecal{false};
    std::unordered_map<std::string, std::unique_ptr<PlaybackPublisherBase>> publishers;
    std::optional<uint64_t> first_timestamp;
    std::chrono::steady_clock::time_point start_wall_time{};
};

/**
 * @brief 键盘控制状态，用于处理终端交互
 */
struct KeyboardControl {
    std::atomic<bool> paused{true};
    std::atomic<bool> stop{false};
    std::thread worker;
    bool terminal_configured{false};
    struct termios original {
    };
};

/**
 * @brief 消息统计信息，便于最终输出回放与过滤数据
 */
struct MessageStats {
    size_t pointcloud{0};
    size_t compressed_image{0};
    size_t imu{0};
    size_t pose{0};
    size_t path{0};
    size_t tf{0};
    size_t filtered{0};
    size_t skipped{0};
    size_t window_skipped{0};
};

/**
 * @brief 可复用的 Protobuf 消息缓存，减少重复分配
 */
struct MessageCache {
    FoxglovePointCloud pointcloud;
    FoxgloveCompressedImage compressed_image;
    FoxgloveImu imu;
    FoxglovePoseInFrame pose_in_frame;
    FoxglovePosesInFrame poses_in_frame;
    FoxgloveFrameTransforms frame_transforms;
};

std::atomic<bool> g_interrupted{false};  ///< SIGINT 退出标志

/**
 * @brief 确保对应类型的 Schema 已注册至 MCAP writer
 * @param recorder 录制上下文
 * @param kind 消息类型
 * @return Schema ID
 */
mcap::SchemaId EnsureSchema(RecorderContext& recorder, MessageKind kind)
{
    auto add_schema = [&](auto& cached, const google::protobuf::Descriptor* descriptor) -> mcap::SchemaId {
        if (!cached.has_value()) {
            const std::string schema_name = descriptor ? descriptor->full_name() : "";
            const std::string schema_data = BuildDescriptorSetString(descriptor);
            mcap::Schema schema(schema_name, "protobuf", schema_data);
            if (!recorder.writer) {
                throw std::runtime_error("recorder writer is not initialized");
            }
            recorder.writer->addSchema(schema);
            cached = schema.id;
        }
        return *cached;
    };

    switch (kind) {
        case MessageKind::PointCloud:
            return add_schema(recorder.schemas.pointcloud, FoxglovePointCloud::descriptor());
        case MessageKind::CompressedImage:
            return add_schema(recorder.schemas.compressed_image, FoxgloveCompressedImage::descriptor());
        case MessageKind::Imu:
            return add_schema(recorder.schemas.imu, FoxgloveImu::descriptor());
        case MessageKind::PoseInFrame:
            return add_schema(recorder.schemas.pose_in_frame, FoxglovePoseInFrame::descriptor());
        case MessageKind::PosesInFrame:
            return add_schema(recorder.schemas.poses_in_frame, FoxglovePosesInFrame::descriptor());
        case MessageKind::FrameTransforms:
            return add_schema(recorder.schemas.frame_transforms, FoxgloveFrameTransforms::descriptor());
        case MessageKind::Unsupported:
        default:
            throw std::runtime_error("unsupported message kind for schema");
    }
}

/**
 * @brief 确保特定话题的 channel 存在，否则创建
 * @param recorder 录制上下文
 * @param topic 话题名称
 * @param kind 消息类型
 * @return channel ID
 */
uint16_t EnsureChannel(RecorderContext& recorder, const std::string& topic, MessageKind kind)
{
    auto it = recorder.channel_ids.find(topic);
    if (it != recorder.channel_ids.end()) {
        return it->second;
    }

    const mcap::SchemaId schema_id = EnsureSchema(recorder, kind);
    mcap::Channel channel(topic, "protobuf", schema_id, {});
    if (!recorder.writer) {
        throw std::runtime_error("recorder writer is not initialized");
    }
    recorder.writer->addChannel(channel);

    const uint16_t channel_id = channel.id;
    recorder.channel_ids.emplace(topic, channel_id);
    recorder.sequence_numbers.emplace(topic, 0);
    recorder.next_channel_id = std::max<uint16_t>(recorder.next_channel_id, channel_id + 1);
    return channel_id;
}

/**
 * @brief 写入单条 MCAP 消息
 * @param recorder 录制上下文
 * @param topic 话题名称
 * @param kind 消息类型
 * @param timestamp_ns 时间戳（纳秒）
 * @param data 序列化后的消息数据
 * @return 成功写入返回 true
 */
bool WriteRecord(RecorderContext& recorder, const std::string& topic, MessageKind kind, uint64_t timestamp_ns, const std::string& data)
{
    if (!recorder.enabled || !recorder.writer) {
        return true;
    }

    const uint16_t channel_id = EnsureChannel(recorder, topic, kind);
    const uint32_t sequence = recorder.sequence_numbers[topic]++;

    mcap::Message msg;
    msg.channelId = channel_id;
    msg.sequence = sequence;
    msg.logTime = timestamp_ns;
    msg.publishTime = timestamp_ns;
    msg.data = reinterpret_cast<const std::byte*>(data.data());
    msg.dataSize = static_cast<uint64_t>(data.size());

    const auto status = recorder.writer->write(msg);
    if (!status.ok()) {
        spdlog::error("Failed to write message for topic {}: {}", topic, status.message);
        return false;
    }
    return true;
}

/**
 * @brief 创建或查找播放发布器
 * @param ctx 回放上下文
 * @param topic_settings 话题配置
 * @param kind 消息类型
 * @return 发布器指针
 */
PlaybackPublisherBase* EnsurePublisher(PlaybackContext& ctx, const TopicSettings& topic_settings, MessageKind kind)
{
    auto it = ctx.publishers.find(topic_settings.publish_service);
    if (it != ctx.publishers.end()) {
        return it->second.get();
    }

    std::unique_ptr<PlaybackPublisherBase> publisher;

    switch (kind) {
        case MessageKind::PointCloud:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePointCloud>>(topic_settings.publish_service);
            break;
        case MessageKind::CompressedImage:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveCompressedImage>>(topic_settings.publish_service);
            break;
        case MessageKind::Imu:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveImu>>(topic_settings.publish_service);
            break;
        case MessageKind::PoseInFrame:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePoseInFrame>>(topic_settings.publish_service);
            break;
        case MessageKind::PosesInFrame:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePosesInFrame>>(topic_settings.publish_service);
            break;
        case MessageKind::FrameTransforms:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveFrameTransforms>>(topic_settings.publish_service);
            break;
        case MessageKind::Unsupported:
        default:
            spdlog::warn("Unsupported message kind for playback on topic {}", topic_settings.publish_service);
            return nullptr;
    }

    auto* raw_ptr = publisher.get();
    ctx.publishers.emplace(topic_settings.publish_service, std::move(publisher));
    return raw_ptr;
}

/**
 * @brief 根据回放速率控制节奏
 * @param ctx 回放上下文
 * @param timestamp_ns 消息时间戳
 */
void PlaybackSleep(PlaybackContext& ctx, uint64_t timestamp_ns)
{
    if (!ctx.enabled || !ctx.sync_time) {
        return;
    }

    if (!ctx.first_timestamp.has_value()) {
        ctx.first_timestamp = timestamp_ns;
        ctx.start_wall_time = std::chrono::steady_clock::now();
        return;
    }

    const uint64_t delta = timestamp_ns - *ctx.first_timestamp;
    const long double scaled = static_cast<long double>(delta) / std::max<long double>(ctx.rate, 1e-6L);
    const auto target_time = ctx.start_wall_time + std::chrono::nanoseconds(static_cast<int64_t>(scaled));
    const auto now = std::chrono::steady_clock::now();
    if (target_time > now) {
        std::this_thread::sleep_for(target_time - now);
    }
}

/**
 * @brief 填充 Protobuf 时间戳
 * @param seconds 秒
 * @param nanoseconds 纳秒
 * @param target 目标时间戳
 */
void FillTimestamp(uint32_t seconds, uint32_t nanoseconds, google::protobuf::Timestamp& target)
{
    target.set_seconds(static_cast<std::int64_t>(seconds));
    target.set_nanos(static_cast<std::int32_t>(nanoseconds));
}

/**
 * @brief 将位姿重置为单位姿态
 * @param pose 目标位姿
 */
void FillIdentityPose(foxglove::Pose& pose)
{
    auto* position = pose.mutable_position();
    position->set_x(0.0);
    position->set_y(0.0);
    position->set_z(0.0);

    auto* orientation = pose.mutable_orientation();
    orientation->set_x(0.0);
    orientation->set_y(0.0);
    orientation->set_z(0.0);
    orientation->set_w(1.0);
}

/**
 * @brief 将 ROS PointField 类型映射到 Foxglove 数值枚举
 * @param datatype ROS 数据类型编码
 * @return 对应的枚举值
 */
foxglove::PackedElementField_NumericType ToNumericType(uint8_t datatype)
{
    switch (datatype) {
        case 1:
            return foxglove::PackedElementField_NumericType_UINT8;
        case 2:
            return foxglove::PackedElementField_NumericType_UINT16;
        case 3:
            return foxglove::PackedElementField_NumericType_UINT32;
        case 4:
            return foxglove::PackedElementField_NumericType_INT8;
        case 5:
            return foxglove::PackedElementField_NumericType_INT16;
        case 6:
            return foxglove::PackedElementField_NumericType_INT32;
        case 7:
            return foxglove::PackedElementField_NumericType_FLOAT32;
        case 8:
            return foxglove::PackedElementField_NumericType_FLOAT64;
        default:
            return foxglove::PackedElementField_NumericType_FLOAT32;
    }
}

/**
 * @brief 解析 MCAP 中的 Protobuf 消息
 * @tparam MessageType 目标消息类型
 * @param view MCAP 消息视图
 * @param message 输出 Protobuf 对象
 * @return 解析成功返回 true
 */
template <typename MessageType>
bool ParseProtobufMessage(const mcap::MessageView& view, MessageType& message)
{
    message.Clear();
    if (view.message.dataSize > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        spdlog::error("Protobuf message too large to parse: {} bytes", view.message.dataSize);
        return false;
    }
    if (!message.ParseFromArray(view.message.data, static_cast<int>(view.message.dataSize))) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse protobuf message on topic {}", topic);
        return false;
    }
    return true;
}

/**
 * @brief 序列化 Protobuf 消息
 * @param message 待序列化的消息
 * @param buffer 输出缓冲
 * @return 成功返回 true
 */
bool SerializeMessage(const google::protobuf::Message& message, std::string& buffer)
{
    buffer.clear();
    if (!message.SerializeToString(&buffer)) {
        spdlog::error("Failed to serialize protobuf message: {}", message.GetTypeName());
        return false;
    }
    return true;
}

/**
 * @brief 将 Livox 点云转换为 Foxglove Protobuf
 * @param view MCAP 消息视图
 * @param message 目标点云消息
 * @return 转换成功返回 true
 */
bool ConvertLivoxPointCloud(const mcap::MessageView& view, FoxglovePointCloud& message)
{
    ROS1LivoxCustomMsg livox_msg;
    if (!livox_msg.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse Livox CustomMsg at topic {}", topic);
        return false;
    }

    constexpr uint32_t kPointStride = 20;
    std::vector<uint8_t> data;
    data.reserve(livox_msg.points.size() * kPointStride);

    auto append_scalar = [&data](auto value) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        data.insert(data.end(), ptr, ptr + sizeof(value));
    };

    for (const auto& point : livox_msg.points) {
        append_scalar(point.x);
        append_scalar(point.y);
        append_scalar(point.z);
        data.push_back(point.reflectivity);
        data.push_back(point.tag);
        data.push_back(point.line);
        data.push_back(0);
        append_scalar(point.offset_time);
    }

    message.Clear();
    FillTimestamp(livox_msg.header.stamp_sec, livox_msg.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(livox_msg.header.frame_id);
    FillIdentityPose(*message.mutable_pose());
    message.set_point_stride(kPointStride);

    message.mutable_fields()->Reserve(7);
    auto* x_field = message.add_fields();
    x_field->set_name("x");
    x_field->set_offset(0);
    x_field->set_type(foxglove::PackedElementField_NumericType_FLOAT32);

    auto* y_field = message.add_fields();
    y_field->set_name("y");
    y_field->set_offset(4);
    y_field->set_type(foxglove::PackedElementField_NumericType_FLOAT32);

    auto* z_field = message.add_fields();
    z_field->set_name("z");
    z_field->set_offset(8);
    z_field->set_type(foxglove::PackedElementField_NumericType_FLOAT32);

    auto* reflectivity_field = message.add_fields();
    reflectivity_field->set_name("reflectivity");
    reflectivity_field->set_offset(12);
    reflectivity_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* tag_field = message.add_fields();
    tag_field->set_name("tag");
    tag_field->set_offset(13);
    tag_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* line_field = message.add_fields();
    line_field->set_name("line");
    line_field->set_offset(14);
    line_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* offset_time_field = message.add_fields();
    offset_time_field->set_name("offset_time");
    offset_time_field->set_offset(16);
    offset_time_field->set_type(foxglove::PackedElementField_NumericType_UINT32);

    message.set_data(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

/**
 * @brief 转换 ROS PointCloud2 或直接解析 Protobuf 点云
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标点云消息
 * @return 转换成功返回 true
 */
bool ConvertPointCloud(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxglovePointCloud& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }
    if (descriptor.is_livox) {
        return ConvertLivoxPointCloud(view, message);
    }

    ROS1PointCloud2 pc2;
    if (!pc2.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/PointCloud2 at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(pc2.header.stamp_sec, pc2.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(pc2.header.frame_id);
    FillIdentityPose(*message.mutable_pose());
    message.set_point_stride(pc2.point_step);

    for (const auto& field : pc2.fields) {
        auto* field_out = message.add_fields();
        field_out->set_name(field.name);
        field_out->set_offset(field.offset);
        field_out->set_type(ToNumericType(field.datatype));
    }

    message.set_data(reinterpret_cast<const char*>(pc2.data.data()), pc2.data.size());
    return true;
}

/**
 * @brief 转换压缩图像或解析 Protobuf 图像
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标图像消息
 * @return 转换成功返回 true
 */
bool ConvertImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxgloveCompressedImage& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1CompressedImage image;
    if (!image.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/CompressedImage at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(image.header.stamp_sec, image.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(image.header.frame_id);
    message.set_format(image.format);
    message.set_data(reinterpret_cast<const char*>(image.data.data()), image.data.size());
    return true;
}

/**
 * @brief 转换 IMU 或解析 Protobuf IMU
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标 IMU 消息
 * @return 转换成功返回 true
 */
bool ConvertImu(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxgloveImu& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1Imu imu;
    if (!imu.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/Imu at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(imu.header.stamp_sec, imu.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(imu.header.frame_id);

    auto* angular_velocity = message.mutable_angular_velocity();
    angular_velocity->set_x(imu.angular_velocity.x);
    angular_velocity->set_y(imu.angular_velocity.y);
    angular_velocity->set_z(imu.angular_velocity.z);

    auto* linear_acceleration = message.mutable_linear_acceleration();
    linear_acceleration->set_x(imu.linear_acceleration.x);
    linear_acceleration->set_y(imu.linear_acceleration.y);
    linear_acceleration->set_z(imu.linear_acceleration.z);
    return true;
}

/**
 * @brief 转换 Path 或解析 Protobuf PosesInFrame
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标轨迹消息
 * @return 转换成功返回 true
 */
bool ConvertPath(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxglovePosesInFrame& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1Path path;
    if (!path.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse nav_msgs/Path at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(path.header.stamp_sec, path.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(path.header.frame_id);

    for (const auto& pose_stamped : path.poses) {
        auto* pose_out = message.add_poses();
        pose_out->mutable_position()->set_x(pose_stamped.pose.position.x);
        pose_out->mutable_position()->set_y(pose_stamped.pose.position.y);
        pose_out->mutable_position()->set_z(pose_stamped.pose.position.z);
        pose_out->mutable_orientation()->set_x(pose_stamped.pose.orientation.x);
        pose_out->mutable_orientation()->set_y(pose_stamped.pose.orientation.y);
        pose_out->mutable_orientation()->set_z(pose_stamped.pose.orientation.z);
        pose_out->mutable_orientation()->set_w(pose_stamped.pose.orientation.w);
    }

    return true;
}

/**
 * @brief 转换 Odometry 或解析 Protobuf PoseInFrame
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标位姿消息
 * @return 转换成功返回 true
 */
bool ConvertOdometry(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxglovePoseInFrame& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1Odometry odom;
    if (!odom.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse nav_msgs/Odometry at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(odom.header.stamp_sec, odom.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(odom.header.frame_id);

    auto* pose = message.mutable_pose();
    pose->mutable_position()->set_x(odom.pose.position.x);
    pose->mutable_position()->set_y(odom.pose.position.y);
    pose->mutable_position()->set_z(odom.pose.position.z);
    pose->mutable_orientation()->set_x(odom.pose.orientation.x);
    pose->mutable_orientation()->set_y(odom.pose.orientation.y);
    pose->mutable_orientation()->set_z(odom.pose.orientation.z);
    pose->mutable_orientation()->set_w(odom.pose.orientation.w);
    return true;
}

/**
 * @brief 转换 TF 消息或解析 Protobuf FrameTransforms
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标 TF 消息
 * @return 转换成功返回 true
 */
bool ConvertTfMessage(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxgloveFrameTransforms& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1TFMessage tf;
    if (!tf.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse tf2_msgs/TFMessage at topic {}", topic);
        return false;
    }

    message.Clear();
    for (const auto& transform_stamped : tf.transforms) {
        auto* tf_out = message.add_transforms();
        FillTimestamp(transform_stamped.header.stamp_sec, transform_stamped.header.stamp_nsec, *tf_out->mutable_timestamp());
        tf_out->set_parent_frame_id(transform_stamped.header.frame_id);
        tf_out->set_child_frame_id(transform_stamped.child_frame_id);
        tf_out->mutable_translation()->set_x(transform_stamped.transform.translation.x);
        tf_out->mutable_translation()->set_y(transform_stamped.transform.translation.y);
        tf_out->mutable_translation()->set_z(transform_stamped.transform.translation.z);
        tf_out->mutable_rotation()->set_x(transform_stamped.transform.rotation.x);
        tf_out->mutable_rotation()->set_y(transform_stamped.transform.rotation.y);
        tf_out->mutable_rotation()->set_z(transform_stamped.transform.rotation.z);
        tf_out->mutable_rotation()->set_w(transform_stamped.transform.rotation.w);
    }

    return true;
}

/**
 * @brief 构建录制输出文件名
 * @param config 工具配置
 * @return 文件路径
 */
std::string BuildOutputFilename(const ToolConfig& config)
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto now_t = system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << config.record_output_dir << '/' << config.record_prefix << '_' << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S") << ".mcap";
    return oss.str();
}

/**
 * @brief 创建录制上下文
 * @param config 工具配置
 * @return 填充后的录制上下文
 */
RecorderContext CreateRecorder(const ToolConfig& config)
{
    RecorderContext recorder;
    if (!config.record_enabled) {
        return recorder;
    }

    std::filesystem::create_directories(config.record_output_dir);
    const std::string output_file = BuildOutputFilename(config);
    if (!config.record_overwrite && std::filesystem::exists(output_file)) {
        throw std::runtime_error("output file already exists: " + output_file);
    }

    auto writer = std::make_unique<mcap::McapWriter>();

    mcap::McapWriterOptions options("protobuf");
    if (config.record_compression == "zstd") {
        options.compression = mcap::Compression::Zstd;
    } else if (config.record_compression == "lz4") {
        options.compression = mcap::Compression::Lz4;
    } else {
        options.compression = mcap::Compression::None;
    }
    options.chunkSize = config.record_chunk_size;

    const auto status = writer->open(output_file, options);
    if (!status.ok()) {
        throw std::runtime_error("failed to open MCAP writer: " + status.message);
    }

    recorder.enabled = true;
    recorder.writer = std::move(writer);
    spdlog::info("Recording enabled -> {}", output_file);
    return recorder;
}

/**
 * @brief 结束录制并关闭 writer
 * @param recorder 录制上下文
 */
void FinalizeRecorder(RecorderContext& recorder)
{
    if (recorder.enabled && recorder.writer) {
        recorder.writer->close();
        recorder.enabled = false;
        recorder.writer.reset();
        spdlog::info("Recording finished");
    }
}

/**
 * @brief 创建回放上下文
 * @param config 工具配置
 * @return 填充后的回放上下文
 */
PlaybackContext CreatePlaybackContext(const ToolConfig& config)
{
    PlaybackContext ctx;
    ctx.enabled = config.playback_enabled;
    ctx.sync_time = config.playback_sync_time;
    ctx.rate = config.playback_rate <= 0.0 ? 1.0 : config.playback_rate;
    if (!ctx.enabled) {
        return ctx;
    }

    const int init_code = eCAL::Initialize("slam_recorder_bag_tool");
    if (init_code < 0) {
        spdlog::error("Failed to initialize eCAL, playback disabled (code {})", init_code);
        ctx.enabled = false;
        return ctx;
    }

    ctx.ecal_initialized = true;
    ctx.owns_ecal = (init_code == 0);
    if (init_code != 0) {
        spdlog::warn("eCAL already initialized, reuse existing context (code {})", init_code);
    }

    spdlog::info("Playback enabled (rate {:.3f}, sync_time={})", ctx.rate, ctx.sync_time);
    return ctx;
}

/**
 * @brief 启动键盘控制线程，实现暂停/恢复交互
 * @param control 键盘控制句柄
 */
void StartKeyboardControl(KeyboardControl& control)
{
    if (!isatty(STDIN_FILENO)) {
        spdlog::warn("Standard input is not a terminal; keyboard control disabled");
        control.paused.store(false);
        return;
    }

    if (::tcgetattr(STDIN_FILENO, &control.original) != 0) {
        spdlog::warn("Failed to read terminal attributes; keyboard control disabled");
        control.paused.store(false);
        return;
    }

    termios raw = control.original;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
        spdlog::warn("Failed to set terminal to raw mode; keyboard control disabled");
        control.paused.store(false);
        return;
    }
    control.terminal_configured = true;

    control.worker = std::thread([&control]() {
        spdlog::info("Keyboard control: press space to toggle playback");
        while (!control.stop.load()) {
            pollfd fd{STDIN_FILENO, POLLIN, 0};
            const int ret = ::poll(&fd, 1, 100);
            if (ret > 0 && (fd.revents & POLLIN)) {
                char ch = 0;
                if (::read(STDIN_FILENO, &ch, 1) > 0) {
                    if (ch == ' ') {
                        const bool paused = !control.paused.load();
                        control.paused.store(paused);
                        if (paused) {
                            spdlog::info("Playback paused");
                        } else {
                            spdlog::info("Playback resumed");
                        }
                    }
                }
            }
        }
    });
}

/**
 * @brief 停止键盘控制线程并恢复终端配置
 * @param control 键盘控制句柄
 */
void StopKeyboardControl(KeyboardControl& control)
{
    control.stop.store(true);
    if (control.worker.joinable()) {
        control.worker.join();
    }
    if (control.terminal_configured) {
        ::tcsetattr(STDIN_FILENO, TCSANOW, &control.original);
        control.terminal_configured = false;
    }
}

/**
 * @brief 将消息发布到回放服务
 * @param ctx 回放上下文
 * @param topic_settings 话题配置
 * @param kind 消息类型
 * @param timestamp_ns 消息时间戳
 * @param message Protobuf 消息体
 */
void PublishPlayback(
    PlaybackContext& ctx,
    const TopicSettings& topic_settings,
    MessageKind kind,
    uint64_t timestamp_ns,
    const google::protobuf::Message& message)
{
    if (!ctx.enabled || !topic_settings.playback) {
        return;
    }

    PlaybackSleep(ctx, timestamp_ns);
    auto* publisher = EnsurePublisher(ctx, topic_settings, kind);
    if (!publisher) {
        return;
    }

    if (!publisher->Publish(message)) {
        spdlog::warn("Failed to publish message on {}", topic_settings.publish_service);
    }
}

/**
 * @brief 运行期上下文，封装录制与回放状态
 */
struct ToolRuntime {
    ToolConfig config;
    RecorderContext recorder;
    PlaybackContext playback;
};

/**
 * @brief 执行 bag_tool 主流程
 * @param config 工具配置
 * @return 成功返回 0，失败返回非零
 */
int RunToolImpl(const ToolConfig& config)
{
    if (config.input_type == InputType::Rosbag) {
        spdlog::error("rosbag inputs are not supported yet. Please convert to MCAP.");
        return 1;
    }

    spdlog::info("===== bag_tool start =====");
    spdlog::info("Input file: {}", config.input_path);
    spdlog::info(
        "Playback: {} (rate {:.2f}, sync {})",
        config.playback_enabled ? "enabled" : "disabled",
        config.playback_rate,
        config.playback_sync_time ? "on" : "off");
    spdlog::info("Recording: {}", config.record_enabled ? "enabled" : "disabled");
    const double sanitized_start_seconds = std::max(0.0, config.processing_start_seconds);
    const bool has_duration = config.processing_duration_seconds > 0.0;
    if (sanitized_start_seconds > 0.0 && has_duration) {
        spdlog::info("Processing window: start {:.3f}s, duration {:.3f}s", sanitized_start_seconds, config.processing_duration_seconds);
    } else if (sanitized_start_seconds > 0.0) {
        spdlog::info("Processing window: start {:.3f}s, until file end", sanitized_start_seconds);
    } else if (has_duration) {
        spdlog::info("Processing window: from beginning, duration {:.3f}s", config.processing_duration_seconds);
    } else {
        spdlog::info("Processing window: whole file");
    }
    constexpr double kNsecPerSec = 1'000'000'000.0;
    const uint64_t start_offset_ns = static_cast<uint64_t>(std::llround(sanitized_start_seconds * kNsecPerSec));
    const uint64_t duration_ns = has_duration ? static_cast<uint64_t>(std::llround(config.processing_duration_seconds * kNsecPerSec)) : 0U;

    mcap::McapReader reader;
    const auto status = reader.open(config.input_path);
    if (!status.ok()) {
        spdlog::error("Failed to open MCAP file {}: {}", config.input_path, status.message);
        return 1;
    }

    uint64_t total_messages = 0;
    const auto summary_status = reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
    if (!summary_status.ok()) {
        spdlog::debug("Failed to read MCAP summary: {}", summary_status.message);
    } else if (const auto stats_opt = reader.statistics(); stats_opt.has_value()) {
        total_messages = stats_opt->messageCount;
    }
    if (total_messages > 0) {
        spdlog::info("Estimated messages: {}", total_messages);
    } else {
        spdlog::info("Estimated messages: unknown");
    }

    ToolRuntime runtime{config, CreateRecorder(config), CreatePlaybackContext(config)};
    MessageStats stats;
    MessageCache message_cache;
    std::string serialized_buffer;
    serialized_buffer.reserve(1024 * 1024);  // 预留缓冲区降低重复分配
    KeyboardControl keyboard;
    StartKeyboardControl(keyboard);
    if (keyboard.paused.load()) {
        spdlog::info("Playback paused. Press space to start");
    }

    auto messages = reader.readMessages();
    uint64_t processed = 0;
    uint64_t next_progress = total_messages > 0 ? std::max<uint64_t>(1, total_messages / 20) : 1000;
    std::optional<std::chrono::steady_clock::time_point> pause_started;
    std::optional<uint64_t> bag_start_time_ns;
    std::optional<uint64_t> processing_start_time_ns;
    std::optional<uint64_t> processing_end_time_ns;
    for (auto it = messages.begin(); it != messages.end(); ++it) {
        const auto& view = *it;
        const std::string topic = view.channel ? view.channel->topic : "";

        if (keyboard.paused.load()) {
            if (!pause_started.has_value()) {
                pause_started = std::chrono::steady_clock::now();
            }
        }
        while (keyboard.paused.load() && !keyboard.stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (keyboard.stop.load()) {
            break;
        }
        if (pause_started.has_value()) {
            if (runtime.playback.first_timestamp.has_value()) {
                runtime.playback.start_wall_time += std::chrono::steady_clock::now() - *pause_started;
            }
            pause_started.reset();
        }

        const uint64_t timestamp_ns = view.message.logTime != 0 ? view.message.logTime : view.message.publishTime;
        // 部分数据可能缺失 logTime，此时回落到 publishTime 保证时间戳连续
        if (!bag_start_time_ns.has_value()) {
            // 初始化bag首帧时间，并依据配置计算处理窗口边界
            bag_start_time_ns = timestamp_ns;
            processing_start_time_ns = bag_start_time_ns.value() + start_offset_ns;
            if (!has_duration) {
                processing_end_time_ns.reset();
            } else {
                processing_end_time_ns = processing_start_time_ns.value() + duration_ns;
            }
        }
        if (processing_start_time_ns.has_value() && timestamp_ns < processing_start_time_ns.value()) {
            // 落在配置窗口左侧的消息直接跳过
            ++stats.window_skipped;
            continue;
        }
        if (processing_end_time_ns.has_value() && timestamp_ns > processing_end_time_ns.value()) {
            // 超出窗口终点后立即终止遍历，避免不必要的解码
            spdlog::info("Processing window reached end timestamp, stopping");
            break;
        }
        if (g_interrupted.load(std::memory_order_relaxed)) {
            spdlog::warn("SIGINT received, stopping gracefully...");
            break;
        }

        const MessageDescriptor descriptor = DescribeMessage(view);
        if (descriptor.kind == MessageKind::Unsupported) {
            ++stats.skipped;
            continue;
        }

        const TopicSettings topic_settings = ResolveTopic(config, topic);
        if (!topic_settings.playback && !topic_settings.record) {
            ++stats.filtered;
            continue;
        }

        google::protobuf::Message* message_ptr = nullptr;
        bool conversion_ok = false;

        switch (descriptor.kind) {
            case MessageKind::PointCloud:
                conversion_ok = ConvertPointCloud(view, descriptor, message_cache.pointcloud);
                message_ptr = &message_cache.pointcloud;
                break;
            case MessageKind::CompressedImage:
                conversion_ok = ConvertImage(view, descriptor, message_cache.compressed_image);
                message_ptr = &message_cache.compressed_image;
                break;
            case MessageKind::Imu:
                conversion_ok = ConvertImu(view, descriptor, message_cache.imu);
                message_ptr = &message_cache.imu;
                break;
            case MessageKind::PoseInFrame:
                conversion_ok = ConvertOdometry(view, descriptor, message_cache.pose_in_frame);
                message_ptr = &message_cache.pose_in_frame;
                break;
            case MessageKind::PosesInFrame:
                conversion_ok = ConvertPath(view, descriptor, message_cache.poses_in_frame);
                message_ptr = &message_cache.poses_in_frame;
                break;
            case MessageKind::FrameTransforms:
                conversion_ok = ConvertTfMessage(view, descriptor, message_cache.frame_transforms);
                message_ptr = &message_cache.frame_transforms;
                break;
            case MessageKind::Unsupported:
            default:
                conversion_ok = false;
                break;
        }

        if (!conversion_ok || message_ptr == nullptr) {
            ++stats.skipped;
            continue;
        }

        if (topic_settings.record) {
            if (descriptor.is_protobuf) {
                serialized_buffer.assign(reinterpret_cast<const char*>(view.message.data), view.message.dataSize);
            } else {
                if (!SerializeMessage(*message_ptr, serialized_buffer)) {
                    ++stats.skipped;
                    continue;
                }
            }
            WriteRecord(runtime.recorder, topic, descriptor.kind, timestamp_ns, serialized_buffer);
        }

        PublishPlayback(runtime.playback, topic_settings, descriptor.kind, timestamp_ns, *message_ptr);

        switch (descriptor.kind) {
            case MessageKind::PointCloud:
                ++stats.pointcloud;
                break;
            case MessageKind::CompressedImage:
                ++stats.compressed_image;
                break;
            case MessageKind::Imu:
                ++stats.imu;
                break;
            case MessageKind::PoseInFrame:
                ++stats.pose;
                break;
            case MessageKind::PosesInFrame:
                ++stats.path;
                break;
            case MessageKind::FrameTransforms:
                ++stats.tf;
                break;
            case MessageKind::Unsupported:
                break;
        }

        ++processed;
        if (total_messages > 0) {
            if (processed >= next_progress || processed == total_messages) {
                double percent = static_cast<double>(processed) * 100.0 / static_cast<double>(total_messages);
                spdlog::info("Progress: {}/{} ({:.1f}%)", processed, total_messages, percent);
                next_progress = std::min(total_messages, processed + std::max<uint64_t>(1, total_messages / 20));
            }
        } else if (processed % 1000 == 0) {
            spdlog::info("Processed {} messages...", processed);
        }
    }

    reader.close();
    FinalizeRecorder(runtime.recorder);
    StopKeyboardControl(keyboard);
    if (runtime.playback.ecal_initialized && runtime.playback.owns_ecal) {
        eCAL::Finalize();
    }

    spdlog::info("PointCloud messages: {}", stats.pointcloud);
    spdlog::info("CompressedImage messages: {}", stats.compressed_image);
    spdlog::info("IMU messages: {}", stats.imu);
    spdlog::info("Pose messages: {}", stats.pose);
    spdlog::info("Path messages: {}", stats.path);
    spdlog::info("FrameTransforms messages: {}", stats.tf);
    spdlog::info("Filtered messages: {}", stats.filtered);
    spdlog::info("Skipped messages: {}", stats.skipped);
    spdlog::info("Time window skipped messages: {}", stats.window_skipped);
    spdlog::info("===== bag_tool finished =====");
    return 0;
}

}  // namespace

/**
 * @brief 从 YAML 配置文件加载 bag_tool 设置
 * @param yaml_path 配置文件绝对路径或相对路径
 * @return 解析后的工具配置
 * @note 解析失败时会抛出 std::runtime_error 异常
 */
ToolConfig LoadBagToolConfig(const std::string& yaml_path)
{
    spdlog::info("Loading bag_tool configuration from {}", yaml_path);
    YAML::Node root = YAML::LoadFile(yaml_path);
    auto bag_node = root["BagTool"];
    if (!bag_node) {
        throw std::runtime_error("BagTool section missing in config");
    }

    ToolConfig config;
    const auto input_node = bag_node["input"];
    if (!input_node) {
        throw std::runtime_error("BagTool.input section is required");
    }
    config.input_path = input_node["path"].as<std::string>();
    const std::string type_str = input_node["type"] ? input_node["type"].as<std::string>() : "ros1_mcap";
    config.input_type = ParseInputType(type_str);

    const auto playback_node = bag_node["playback"];
    if (playback_node) {
        // 回放配置允许缺省，优先使用配置文件中的覆盖值
        config.playback_enabled = playback_node["enable"].as<bool>(false);
        config.playback_rate = playback_node["rate"].as<double>(1.0);
        config.playback_sync_time = playback_node["synchronize_time"].as<bool>(true);
    }

    const auto window_node = bag_node["time_window"];
    if (window_node) {
        config.processing_start_seconds = window_node["start_seconds"].as<double>(config.processing_start_seconds);
        config.processing_duration_seconds = window_node["duration_seconds"].as<double>(config.processing_duration_seconds);
    }

    const auto record_node = bag_node["record"];
    if (record_node) {
        config.record_enabled = record_node["enable"].as<bool>(false);
        config.record_output_dir = record_node["output_dir"].as<std::string>(config.record_output_dir);
        config.record_prefix = record_node["filename_prefix"].as<std::string>(config.record_prefix);
        config.record_compression = record_node["compression"].as<std::string>(config.record_compression);
        config.record_chunk_size = record_node["chunk_size"].as<uint64_t>(config.record_chunk_size);
        config.record_overwrite = record_node["overwrite_existing"].as<bool>(config.record_overwrite);
    }

    const auto topics_node = bag_node["topics"];
    if (topics_node) {
        // 提供全局默认值后，再读取具体的话题设置
        config.default_playback = topics_node["default_playback_enabled"].as<bool>(true);
        config.default_record = topics_node["default_record_enabled"].as<bool>(true);

        const auto entries = topics_node["entries"];
        if (entries && entries.IsSequence()) {
            for (const auto& entry : entries) {
                TopicSettings settings;
                const std::string name = entry["name"].as<std::string>();
                if (entry["publish_service"]) {
                    settings.publish_service = entry["publish_service"].as<std::string>();
                } else {
                    settings.publish_service = name;
                }
                settings.playback = entry["playback_enabled"].as<bool>(true);
                settings.record = entry["record_enabled"].as<bool>(true);
                config.topics.emplace(name, settings);
            }
        }
    }

    return config;
}

/**
 * @brief 对外暴露的工具入口
 * @param config 工具配置
 * @return 运行结果码
 */
int RunTool(const ToolConfig& config)
{
    return RunToolImpl(config);
}

}  // namespace ms_slam::slam_recorder

/**
 * @brief 程序主入口
 * @param argc 参数数量
 * @param argv 参数数组
 * @return 进程退出码
 */
int main(int argc, char** argv)
{
    try {
        std::signal(SIGINT, [](int) {
            ms_slam::slam_recorder::g_interrupted.store(true, std::memory_order_relaxed);
            spdlog::warn("SIGINT captured, finishing current writes then exit");
        });

        const std::string config_path = (argc > 1) ? argv[1] : "../config/config.yaml";
        auto logger = spdlog::get("bag_tool");
        if (!logger) {
            logger = spdlog::stdout_color_mt("bag_tool");
        }
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);

        auto config = ms_slam::slam_recorder::LoadBagToolConfig(config_path);
        return ms_slam::slam_recorder::RunTool(config);
    } catch (const std::exception& ex) {
        spdlog::critical("bag_tool failed: {}", ex.what());
        return 1;
    }
}
