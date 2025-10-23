#include "slam_recorder/bag_tool.hpp"
#include "slam_recorder/ros1_msg.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <flatbuffers/flatbuffers.h>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/PoseInFrame_generated.h>
#include <fbs/PosesInFrame_generated.h>
#include <iox2/iceoryx2.hpp>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

namespace ms_slam::slam_recorder
{

namespace
{

using ms_slam::slam_common::FBSPublisher;
using ms_slam::slam_common::FoxgloveCompressedImage;
using ms_slam::slam_common::FoxgloveImu;
using ms_slam::slam_common::FoxglovePointCloud;
using ms_slam::slam_common::FoxglovePoseInFrame;
using ms_slam::slam_common::FoxglovePosesInFrame;

enum class MessageKind { PointCloud, CompressedImage, Imu, PoseInFrame, PosesInFrame, Unsupported };

struct MessageDescriptor {
    MessageKind kind{MessageKind::Unsupported};
    bool is_ros{false};
    bool is_flatbuffer{false};
    bool is_livox{false};
};

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

InputType parse_input_type(const std::string& value)
{
    const auto lowered = to_lower(value);
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

MessageDescriptor describe_message(const mcap::MessageView& view)
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
    }

    descriptor.kind = MessageKind::Unsupported;
    return descriptor;
}

TopicSettings resolve_topic(const ToolConfig& config, const std::string& topic)
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

struct SchemaCache {
    std::optional<mcap::SchemaId> pointcloud;
    std::optional<mcap::SchemaId> compressed_image;
    std::optional<mcap::SchemaId> imu;
    std::optional<mcap::SchemaId> pose_in_frame;
    std::optional<mcap::SchemaId> poses_in_frame;
};

struct RecorderContext {
    bool enabled{false};
    std::unique_ptr<mcap::McapWriter> writer;
    SchemaCache schemas;
    std::unordered_map<std::string, uint16_t> channel_ids;
    std::unordered_map<std::string, uint32_t> sequence_numbers;
    uint16_t next_channel_id{1};
};

struct PlaybackPublisherBase {
    virtual ~PlaybackPublisherBase() = default;
    virtual bool publish(const std::byte* data, size_t size) = 0;
};

template <typename MessageType>
class PlaybackPublisher final : public PlaybackPublisherBase
{
  public:
    PlaybackPublisher(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name, uint32_t queue_depth)
    : publisher_(std::move(node), service_name, slam_common::PubSubConfig{.subscriber_max_buffer_size = queue_depth})
    {
    }

    bool publish(const std::byte* data, size_t size) override { return publisher_.publish_raw(reinterpret_cast<const uint8_t*>(data), size); }

  private:
    FBSPublisher<MessageType> publisher_;
};

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

struct KeyboardControl {
    std::atomic<bool> paused{true};
    std::atomic<bool> stop{false};
    std::thread worker;
    bool terminal_configured{false};
    struct termios original {
    };
};

struct MessageStats {
    size_t pointcloud{0};
    size_t compressed_image{0};
    size_t imu{0};
    size_t pose{0};
    size_t path{0};
    size_t filtered{0};
    size_t skipped{0};
};

mcap::SchemaId ensure_schema(RecorderContext& recorder, MessageKind kind)
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
        case MessageKind::Unsupported:
        default:
            throw std::runtime_error("unsupported message kind for schema");
    }
}

uint16_t ensure_channel(RecorderContext& recorder, const std::string& topic, MessageKind kind)
{
    auto it = recorder.channel_ids.find(topic);
    if (it != recorder.channel_ids.end()) {
        return it->second;
    }

    const mcap::SchemaId schema_id = ensure_schema(recorder, kind);
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

bool write_record(RecorderContext& recorder, const std::string& topic, MessageKind kind, uint64_t timestamp_ns, const std::byte* data, size_t size)
{
    if (!recorder.enabled || !recorder.writer) {
        return true;
    }

    const uint16_t channel_id = ensure_channel(recorder, topic, kind);
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

PlaybackPublisherBase* ensure_publisher(PlaybackContext& ctx, const TopicSettings& topic_settings, MessageKind kind)
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
        case MessageKind::Unsupported:
        default:
            spdlog::warn("Unsupported message kind for playback on topic {}", topic_settings.publish_service);
            return nullptr;
    }

    auto* raw_ptr = publisher.get();
    ctx.publishers.emplace(topic_settings.publish_service, std::move(publisher));
    return raw_ptr;
}

void playback_sleep(PlaybackContext& ctx, uint64_t timestamp_ns)
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

bool convert_pointcloud(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1PointCloud2 pc2;
    if (!pc2.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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
        fields.emplace_back(foxglove::CreatePackedElementField(fbb, fbb.CreateString(field.name), field.offset, numeric_type));
    }

    auto data_vector = fbb.CreateVector(pc2.data);
    auto fields_vector = fbb.CreateVector(fields);

    auto pointcloud = foxglove::CreatePointCloud(fbb, &timestamp, frame_id, 0, pc2.point_step, fields_vector, data_vector);
    fbb.Finish(pointcloud);
    return true;
}

bool convert_livox_pointcloud(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1LivoxCustomMsg livox_msg;
    if (!livox_msg.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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

bool convert_image(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1CompressedImage image;
    if (!image.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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

bool convert_imu(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Imu imu;
    if (!imu.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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

bool convert_path(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Path path;
    if (!path.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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

bool convert_odometry(const mcap::MessageView& view, flatbuffers::FlatBufferBuilder& fbb)
{
    ROS1Odometry odom;
    if (!odom.parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
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

bool convert_ros_message(const mcap::MessageView& view, const MessageDescriptor& descriptor, flatbuffers::FlatBufferBuilder& fbb)
{
    switch (descriptor.kind) {
        case MessageKind::PointCloud:
            return descriptor.is_livox ? convert_livox_pointcloud(view, fbb) : convert_pointcloud(view, fbb);
        case MessageKind::CompressedImage:
            return convert_image(view, fbb);
        case MessageKind::Imu:
            return convert_imu(view, fbb);
        case MessageKind::PoseInFrame:
            return convert_odometry(view, fbb);
        case MessageKind::PosesInFrame:
            return convert_path(view, fbb);
        case MessageKind::Unsupported:
        default:
            return false;
    }
}

std::string build_output_filename(const ToolConfig& config)
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto now_t = system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << config.record_output_dir << '/' << config.record_prefix << '_' << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S") << ".mcap";
    return oss.str();
}

RecorderContext create_recorder(const ToolConfig& config)
{
    RecorderContext recorder;
    if (!config.record_enabled) {
        return recorder;
    }

    std::filesystem::create_directories(config.record_output_dir);
    const std::string output_file = build_output_filename(config);
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

void finalize_recorder(RecorderContext& recorder)
{
    if (recorder.enabled && recorder.writer) {
        recorder.writer->close();
        recorder.enabled = false;
        recorder.writer.reset();
        spdlog::info("Recording finished");
    }
}

PlaybackContext create_playback_context(const ToolConfig& config)
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

void start_keyboard_control(KeyboardControl& control)
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

void stop_keyboard_control(KeyboardControl& control)
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

void publish_playback(
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

    playback_sleep(ctx, timestamp_ns);
    auto* publisher = ensure_publisher(ctx, topic_settings, kind);
    if (!publisher) {
        return;
    }

    if (!publisher->publish(data, size)) {
        spdlog::warn("Failed to publish message on {}", topic_settings.publish_service);
    }
}

struct ToolRuntime {
    ToolConfig config;
    RecorderContext recorder;
    PlaybackContext playback;
};

int run_tool_impl(const ToolConfig& config)
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

    ToolRuntime runtime{config, create_recorder(config), create_playback_context(config)};
    MessageStats stats;
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);
    KeyboardControl keyboard;
    start_keyboard_control(keyboard);
    if (keyboard.paused.load()) {
        spdlog::info("Playback paused. Press space to start");
    }

    auto messages = reader.readMessages();
    uint64_t processed = 0;
    uint64_t next_progress = total_messages > 0 ? std::max<uint64_t>(1, total_messages / 20) : 1000;
    std::optional<std::chrono::steady_clock::time_point> pause_started;
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

        const MessageDescriptor descriptor = describe_message(view);
        if (descriptor.kind == MessageKind::Unsupported) {
            ++stats.skipped;
            continue;
        }

        const TopicSettings topic_settings = resolve_topic(config, topic);
        if (!topic_settings.playback && !topic_settings.record) {
            ++stats.filtered;
            continue;
        }

        const uint64_t timestamp_ns = view.message.logTime != 0 ? view.message.logTime : view.message.publishTime;
        const std::byte* payload = nullptr;
        size_t payload_size = 0;

        if (descriptor.is_flatbuffer) {
            payload = view.message.data;
            payload_size = view.message.dataSize;
        } else if (descriptor.is_ros) {
            if (!convert_ros_message(view, descriptor, fbb)) {
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
            write_record(runtime.recorder, topic, descriptor.kind, timestamp_ns, payload, payload_size);
        }

        publish_playback(runtime.playback, topic_settings, descriptor.kind, timestamp_ns, payload, payload_size);

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
    finalize_recorder(runtime.recorder);
    stop_keyboard_control(keyboard);

    spdlog::info("PointCloud messages: {}", stats.pointcloud);
    spdlog::info("CompressedImage messages: {}", stats.compressed_image);
    spdlog::info("IMU messages: {}", stats.imu);
    spdlog::info("Pose messages: {}", stats.pose);
    spdlog::info("Path messages: {}", stats.path);
    spdlog::info("Filtered messages: {}", stats.filtered);
    spdlog::info("Skipped messages: {}", stats.skipped);
    spdlog::info("===== bag_tool finished =====");
    return 0;
}

}  // namespace

ToolConfig load_bag_tool_config(const std::string& yaml_path)
{
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
    config.input_type = parse_input_type(type_str);

    const auto playback_node = bag_node["playback"];
    if (playback_node) {
        config.playback_enabled = playback_node["enable"].as<bool>(false);
        config.playback_rate = playback_node["rate"].as<double>(1.0);
        config.playback_sync_time = playback_node["synchronize_time"].as<bool>(true);
        config.playback_queue_depth = playback_node["default_queue_depth"].as<uint32_t>(10);
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

int run_tool(const ToolConfig& config)
{
    return run_tool_impl(config);
}

}  // namespace ms_slam::slam_recorder

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

        auto config = ms_slam::slam_recorder::load_bag_tool_config(config_path);
        return ms_slam::slam_recorder::run_tool(config);
    } catch (const std::exception& ex) {
        spdlog::critical("bag_tool failed: {}", ex.what());
        return 1;
    }
}
