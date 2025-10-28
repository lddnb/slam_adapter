#include "slam_recorder/bag_tool.hpp"

#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <iox2/iceoryx2.hpp>

#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

#include "slam_recorder/ros1_msg.hpp"

namespace ms_slam::slam_recorder
{

namespace
{

using ms_slam::slam_common::FBSPublisher;
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
    bool is_flatbuffer{false};
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
    if (lowered == "flatbuffer_mcap" || lowered == "flatbuffers_mcap" || lowered == "foxglove_mcap") {
        return InputType::FlatbufferMcap;
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

    if (encoding == "flatbuffer") {
        if (schema_name == "foxglove.PointCloud") {
            descriptor.kind = MessageKind::PointCloud;
            descriptor.is_flatbuffer = true;
            return descriptor;
        }
        if (schema_name == "foxglove.CompressedImage") {
            descriptor.kind = MessageKind::CompressedImage;
            descriptor.is_flatbuffer = true;
            return descriptor;
        }
        if (schema_name == "foxglove.Imu") {
            descriptor.kind = MessageKind::Imu;
            descriptor.is_flatbuffer = true;
            return descriptor;
        }
        if (schema_name == "foxglove.PoseInFrame") {
            descriptor.kind = MessageKind::PoseInFrame;
            descriptor.is_flatbuffer = true;
            return descriptor;
        }
        if (schema_name == "foxglove.PosesInFrame") {
            descriptor.kind = MessageKind::PosesInFrame;
            descriptor.is_flatbuffer = true;
            return descriptor;
        }
        if (schema_name == "foxglove.FrameTransforms") {
            descriptor.kind = MessageKind::FrameTransforms;
            descriptor.is_flatbuffer = true;
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
    resolved.queue_depth = config.playback_queue_depth;

    const auto it = config.topics.find(topic);
    if (it != config.topics.end()) {
        if (!it->second.publish_service.empty()) {
            resolved.publish_service = it->second.publish_service;
        }
        resolved.queue_depth = it->second.queue_depth == 0 ? config.playback_queue_depth : it->second.queue_depth;
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
 * @brief 回放发布器基类，提供统一的 raw publish 接口
 */
struct PlaybackPublisherBase {
    virtual ~PlaybackPublisherBase() = default;
    virtual bool Publish(const std::byte* data, size_t size) = 0;
};

/**
 * @brief 模板化的回放发布器，实现针对不同消息类型的 raw publish
 * @tparam MessageType FlatBuffer 消息类型
 */
template <typename MessageType>
class PlaybackPublisher final : public PlaybackPublisherBase
{
  public:
    /**
     * @brief 构造函数
     * @param node iceoryx2 节点
     * @param service_name 服务名称
     * @param queue_depth 队列深度
     */
    PlaybackPublisher(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name, uint32_t queue_depth)
    : publisher_(std::move(node), service_name, slam_common::PubSubConfig{.subscriber_max_buffer_size = queue_depth})
    {
    }

    /**
     * @brief 发布原始 FlatBuffer 数据
     * @param data 数据指针
     * @param size 数据长度
     * @return 发布成功返回 true
     */
    bool Publish(const std::byte* data, size_t size) override { return publisher_.publish_raw(reinterpret_cast<const uint8_t*>(data), size); }

  private:
    FBSPublisher<MessageType> publisher_;
};

/**
 * @brief 回放上下文，包含调度、节点及发布器缓存
 */
struct PlaybackContext {
    bool enabled{false};
    bool sync_time{true};
    double rate{1.0};
    uint32_t default_queue_depth{10};
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node;
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
 * @brief 确保对应类型的 Schema 已注册至 MCAP writer
 * @param recorder 录制上下文
 * @param kind 消息类型
 * @return Schema ID
 */
mcap::SchemaId EnsureSchema(RecorderContext& recorder, MessageKind kind)
{
    auto add_schema = [&](auto& cached, auto&& schema_factory) -> mcap::SchemaId {
        if (!cached.has_value()) {
            mcap::Schema schema = schema_factory();
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
            return add_schema(recorder.schemas.pointcloud, [] {
                foxglove::PointCloudBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.PointCloud",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
        case MessageKind::CompressedImage:
            return add_schema(recorder.schemas.compressed_image, [] {
                foxglove::CompressedImageBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.CompressedImage",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
        case MessageKind::Imu:
            return add_schema(recorder.schemas.imu, [] {
                foxglove::ImuBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.Imu",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
        case MessageKind::PoseInFrame:
            return add_schema(recorder.schemas.pose_in_frame, [] {
                foxglove::PoseInFrameBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.PoseInFrame",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
        case MessageKind::PosesInFrame:
            return add_schema(recorder.schemas.poses_in_frame, [] {
                foxglove::PosesInFrameBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.PosesInFrame",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
        case MessageKind::FrameTransforms:
            return add_schema(recorder.schemas.frame_transforms, [] {
                foxglove::FrameTransformsBinarySchema schema_buffer;
                return mcap::Schema(
                    "foxglove.FrameTransforms",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(schema_buffer.data()), schema_buffer.size()));
            });
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
    mcap::Channel channel(topic, "flatbuffer", schema_id, {});
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
 * @param data 消息数据指针
 * @param size 数据大小
 * @return 成功写入返回 true
 */
bool WriteRecord(RecorderContext& recorder, const std::string& topic, MessageKind kind, uint64_t timestamp_ns, const std::byte* data, size_t size)
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
    msg.data = data;
    msg.dataSize = static_cast<uint64_t>(size);

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

    const uint32_t queue_depth = topic_settings.queue_depth == 0 ? ctx.default_queue_depth : topic_settings.queue_depth;
    std::unique_ptr<PlaybackPublisherBase> publisher;

    switch (kind) {
        case MessageKind::PointCloud:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePointCloud>>(ctx.node, topic_settings.publish_service, queue_depth);
            break;
        case MessageKind::CompressedImage:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveCompressedImage>>(ctx.node, topic_settings.publish_service, queue_depth);
            break;
        case MessageKind::Imu:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveImu>>(ctx.node, topic_settings.publish_service, queue_depth);
            break;
        case MessageKind::PoseInFrame:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePoseInFrame>>(ctx.node, topic_settings.publish_service, queue_depth);
            break;
        case MessageKind::PosesInFrame:
            publisher = std::make_unique<PlaybackPublisher<FoxglovePosesInFrame>>(ctx.node, topic_settings.publish_service, queue_depth);
            break;
        case MessageKind::FrameTransforms:
            publisher = std::make_unique<PlaybackPublisher<FoxgloveFrameTransforms>>(ctx.node, topic_settings.publish_service, queue_depth);
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
 * @brief 将 ROS PointCloud2 转换为 Foxglove PointCloud 消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertPointCloud(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1PointCloud2 pc2;
    if (!pc2.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse sensor_msgs/PointCloud2 at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    foxglove::Time timestamp(pc2.header.stamp_sec, pc2.header.stamp_nsec);
    auto frame_id = fbb.CreateString(pc2.header.frame_id);

    std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fields;
    fields.reserve(pc2.fields.size());
    for (const auto& field : pc2.fields) {
        foxglove::NumericType numeric_type;
        switch (field.datatype) {
            case 1:
                numeric_type = foxglove::NumericType_UINT8;
                break;
            case 2:
                numeric_type = foxglove::NumericType_UINT16;
                break;
            case 3:
                numeric_type = foxglove::NumericType_UINT32;
                break;
            case 4:
                numeric_type = foxglove::NumericType_INT8;
                break;
            case 5:
                numeric_type = foxglove::NumericType_INT16;
                break;
            case 6:
                numeric_type = foxglove::NumericType_INT32;
                break;
            case 7:
                numeric_type = foxglove::NumericType_FLOAT32;
                break;
            case 8:
                numeric_type = foxglove::NumericType_FLOAT64;
                break;
            default:
                numeric_type = foxglove::NumericType_FLOAT32;
                break;
        }
        // 将 ROS PointField 元信息映射成 Foxglove 的 PackedElementField 描述
        fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString(field.name), field.offset, numeric_type));
    }

    auto data_vector = fbb.CreateVector(pc2.data);
    auto fields_vector = fbb.CreateVector(fields);

    auto pointcloud = foxglove::CreatePointCloud(fbb, &timestamp, frame_id, 0, pc2.point_step, fields_vector, data_vector);
    fbb.Finish(pointcloud);
    return true;
}

/**
 * @brief 将 Livox 自定义点云转换为 Foxglove PointCloud
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertLivoxPointCloud(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1LivoxCustomMsg livox_msg;
    if (!livox_msg.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse Livox CustomMsg at topic {}", view.channel->topic);
        return false;
    }

    const uint32_t point_step = 20;
    std::vector<uint8_t> data;
    data.reserve(livox_msg.points.size() * point_step);

    auto append_scalar = [&data](auto value) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&value);
        data.insert(data.end(), ptr, ptr + sizeof(value));
    };

    for (const auto& point : livox_msg.points) {
        // Livox 点云字段布局固定，直接串联各标量字段生成连续存储
        append_scalar(point.x);
        append_scalar(point.y);
        append_scalar(point.z);
        data.push_back(point.reflectivity);
        data.push_back(point.tag);
        data.push_back(point.line);
        data.push_back(0);
        append_scalar(point.offset_time);
    }

    fbb.Clear();
    foxglove::Time timestamp(livox_msg.header.stamp_sec, livox_msg.header.stamp_nsec);
    auto frame_id = fbb.CreateString(livox_msg.header.frame_id);

    std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fields;
    fields.reserve(7);
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("x"), 0, foxglove::NumericType_FLOAT32));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("y"), 4, foxglove::NumericType_FLOAT32));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("z"), 8, foxglove::NumericType_FLOAT32));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("reflectivity"), 12, foxglove::NumericType_UINT8));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("tag"), 13, foxglove::NumericType_UINT8));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("line"), 14, foxglove::NumericType_UINT8));
    fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString("offset_time"), 16, foxglove::NumericType_UINT32));

    auto fields_vector = fbb.CreateVector(fields);
    auto data_vector = fbb.CreateVector(data);
    auto pointcloud = foxglove::CreatePointCloud(fbb, &timestamp, frame_id, 0, point_step, fields_vector, data_vector);
    fbb.Finish(pointcloud);
    return true;
}

/**
 * @brief 转换压缩图像消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertImage(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1CompressedImage image;
    if (!image.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse sensor_msgs/CompressedImage at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    foxglove::Time timestamp(image.header.stamp_sec, image.header.stamp_nsec);
    auto frame_id = fbb.CreateString(image.header.frame_id);
    auto format = fbb.CreateString(image.format);
    auto data = fbb.CreateVector(image.data);

    auto compressed = foxglove::CreateCompressedImage(fbb, &timestamp, frame_id, data, format);
    fbb.Finish(compressed);
    return true;
}

/**
 * @brief 转换 IMU 消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertImu(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Imu imu;
    if (!imu.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse sensor_msgs/Imu at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    foxglove::Time timestamp(imu.header.stamp_sec, imu.header.stamp_nsec);
    auto frame_id = fbb.CreateString(imu.header.frame_id);
    auto angular_velocity = foxglove::CreateVector3(fbb, imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
    auto linear_acceleration = foxglove::CreateVector3(fbb, imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);

    auto imu_fb = foxglove::CreateImu(fbb, &timestamp, frame_id, angular_velocity, linear_acceleration);
    fbb.Finish(imu_fb);
    return true;
}

/**
 * @brief 转换 Path 消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertPath(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Path path;
    if (!path.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse nav_msgs/Path at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    foxglove::Time timestamp(path.header.stamp_sec, path.header.stamp_nsec);
    auto frame_id = fbb.CreateString(path.header.frame_id);

    std::vector<flatbuffers::Offset<foxglove::Pose>> pose_offsets;
    pose_offsets.reserve(path.poses.size());
    for (const auto& pose_stamped : path.poses) {
        const auto& pose = pose_stamped.pose;
        auto position = foxglove::CreateVector3(fbb, pose.position.x, pose.position.y, pose.position.z);
        auto orientation = foxglove::CreateQuaternion(fbb, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        pose_offsets.emplace_back(foxglove::CreatePose(fbb, position, orientation));
    }

    auto poses_vector = fbb.CreateVector(pose_offsets);
    auto poses_in_frame = foxglove::CreatePosesInFrame(fbb, &timestamp, frame_id, poses_vector);
    fbb.Finish(poses_in_frame);
    return true;
}

/**
 * @brief 转换 Odometry 消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertOdometry(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Odometry odom;
    if (!odom.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse nav_msgs/Odometry at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    foxglove::Time timestamp(odom.header.stamp_sec, odom.header.stamp_nsec);
    const std::string& frame_source = odom.header.frame_id;
    auto frame_id = fbb.CreateString(frame_source);

    auto position = foxglove::CreateVector3(fbb, odom.pose.position.x, odom.pose.position.y, odom.pose.position.z);
    auto orientation =
        foxglove::CreateQuaternion(fbb, odom.pose.orientation.x, odom.pose.orientation.y, odom.pose.orientation.z, odom.pose.orientation.w);
    auto pose = foxglove::CreatePose(fbb, position, orientation);
    auto pose_in_frame = foxglove::CreatePoseInFrame(fbb, &timestamp, frame_id, pose);
    fbb.Finish(pose_in_frame);
    return true;
}

/**
 * @brief 转换 TF 消息
 * @param view MCAP 消息视图
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertTfMessage(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1TFMessage tf;
    if (!tf.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        spdlog::error("Failed to parse tf2_msgs/TFMessage at topic {}", view.channel->topic);
        return false;
    }

    fbb.Clear();
    std::vector<flatbuffers::Offset<foxglove::FrameTransform>> transform_offsets;
    transform_offsets.reserve(tf.transforms.size());

    for (const auto& transform_stamped : tf.transforms) {
        foxglove::Time timestamp(transform_stamped.header.stamp_sec, transform_stamped.header.stamp_nsec);
        auto parent = fbb.CreateString(transform_stamped.header.frame_id);
        auto child = fbb.CreateString(transform_stamped.child_frame_id);
        auto translation = foxglove::CreateVector3(
            fbb,
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z);
        auto rotation = foxglove::CreateQuaternion(
            fbb,
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z,
            transform_stamped.transform.rotation.w);

        transform_offsets.emplace_back(foxglove::CreateFrameTransform(fbb, &timestamp, parent, child, translation, rotation));
    }

    auto transforms_vector = fbb.CreateVector(transform_offsets);
    auto frame_transforms = foxglove::CreateFrameTransforms(fbb, transforms_vector);
    fbb.Finish(frame_transforms);
    return true;
}

/**
 * @brief 统一处理 ROS 类型消息的转换
 * @param view MCAP 消息视图
 * @param descriptor 消息描述信息
 * @param fbb FlatBufferBuilder 实例
 * @return 转换成功返回 true
 */
bool ConvertRosMessage(const mcap::MessageView& view, const MessageDescriptor& descriptor, flatbuffers::FlatBufferBuilder& fbb)
{
    switch (descriptor.kind) {
        case MessageKind::PointCloud:
            return descriptor.is_livox ? ConvertLivoxPointCloud(view, fbb) : ConvertPointCloud(view, fbb);
        case MessageKind::CompressedImage:
            return ConvertImage(view, fbb);
        case MessageKind::Imu:
            return ConvertImu(view, fbb);
        case MessageKind::PoseInFrame:
            return ConvertOdometry(view, fbb);
        case MessageKind::PosesInFrame:
            return ConvertPath(view, fbb);
        case MessageKind::FrameTransforms:
            return ConvertTfMessage(view, fbb);
        case MessageKind::Unsupported:
        default:
            return false;
    }
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

    mcap::McapWriterOptions options("flatbuffers");
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
    ctx.default_queue_depth = config.playback_queue_depth;
    if (!ctx.enabled) {
        return ctx;
    }

    ctx.node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("create IPC node"));
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
 * @param data 消息数据
 * @param size 数据长度
 */
void PublishPlayback(
    PlaybackContext& ctx,
    const TopicSettings& topic_settings,
    MessageKind kind,
    uint64_t timestamp_ns,
    const std::byte* data,
    size_t size)
{
    if (!ctx.enabled || !topic_settings.playback) {
        return;
    }

    PlaybackSleep(ctx, timestamp_ns);
    auto* publisher = EnsurePublisher(ctx, topic_settings, kind);
    if (!publisher) {
        return;
    }

    if (!publisher->Publish(data, size)) {
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
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);
    // 预分配 1MB 缓冲区，避免频繁扩容带来的重新分配开销
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

        const std::byte* payload = nullptr;
        size_t payload_size = 0;

        // 按消息类型选择直接使用原始数据或先转换再回放
        if (descriptor.is_flatbuffer) {
            payload = view.message.data;
            payload_size = view.message.dataSize;
        } else if (descriptor.is_ros) {
            if (!ConvertRosMessage(view, descriptor, fbb)) {
                ++stats.skipped;
                continue;
            }
            payload = reinterpret_cast<const std::byte*>(fbb.GetBufferPointer());
            payload_size = fbb.GetSize();
        } else {
            ++stats.skipped;
            continue;
        }

        if (topic_settings.record) {
            WriteRecord(runtime.recorder, topic, descriptor.kind, timestamp_ns, payload, payload_size);
        }

        PublishPlayback(runtime.playback, topic_settings, descriptor.kind, timestamp_ns, payload, payload_size);

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
        config.playback_queue_depth = playback_node["default_queue_depth"].as<uint32_t>(10);
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
                settings.queue_depth = entry["queue_depth"].as<uint32_t>(config.playback_queue_depth);
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
