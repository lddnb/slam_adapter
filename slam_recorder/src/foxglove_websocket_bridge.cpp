#include "slam_recorder/foxglove_websocket_bridge.hpp"

#include <cctype>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

namespace ms_slam::slam_recorder
{
namespace
{
using FoxglovePointCloud = ::foxglove::PointCloud;
using FoxgloveCompressedImage = ::foxglove::CompressedImage;
using FoxgloveImu = ::foxglove::Imu;
using FoxglovePoseInFrame = ::foxglove::PoseInFrame;
using FoxglovePosesInFrame = ::foxglove::PosesInFrame;
using FoxgloveFrameTransforms = ::foxglove::FrameTransforms;
using FoxgloveSceneUpdate = ::foxglove::SceneUpdate;

/**
 * @brief 根据 schema 名称获取对应的 Protobuf 描述符
 * @param schema_name Schema 名称
 * @return 描述符指针，未知返回 nullptr
 */
const google::protobuf::Descriptor* ResolveDescriptor(const std::string& schema_name)
{
    if (schema_name == "foxglove.PointCloud") {
        return FoxglovePointCloud::descriptor();
    }
    if (schema_name == "foxglove.CompressedImage") {
        return FoxgloveCompressedImage::descriptor();
    }
    if (schema_name == "foxglove.Imu") {
        return FoxgloveImu::descriptor();
    }
    if (schema_name == "foxglove.PoseInFrame") {
        return FoxglovePoseInFrame::descriptor();
    }
    if (schema_name == "foxglove.PosesInFrame") {
        return FoxglovePosesInFrame::descriptor();
    }
    if (schema_name == "foxglove.FrameTransforms") {
        return FoxgloveFrameTransforms::descriptor();
    }
    if (schema_name == "foxglove.SceneUpdate") {
        return FoxgloveSceneUpdate::descriptor();
    }
    return nullptr;
}

/**
 * @brief 将纳秒时间戳填充到 Timestamp
 */
void FillTimestampFromNs(uint64_t timestamp_ns, google::protobuf::Timestamp& stamp)
{
    constexpr uint64_t kNsPerSec = 1'000'000'000ULL;
    stamp.set_seconds(static_cast<std::int64_t>(timestamp_ns / kNsPerSec));
    stamp.set_nanos(static_cast<std::int32_t>(timestamp_ns % kNsPerSec));
}

/**
 * @brief 将定长字符数组转换为安全字符串
 */
template <std::size_t N>
std::string ToSafeString(const std::array<char, N>& buffer)
{
    return std::string(buffer.data());
}

/**
 * @brief 将 Pose 设置为单位姿态
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
 * @brief 将编码字符串转换为小写形式
 * @param encoding 原始编码
 * @return 小写编码结果
 */
std::string NormalizeEncoding(const std::string& encoding)
{
    std::string lowered = encoding;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

/**
 * @brief 规范化压缩格式，未知值回退到 jpeg
 * @param encoding 输入编码字符串
 * @return 用于 foxglove.CompressedImage 的格式
 */
std::string NormalizeImageFormat(const std::string& encoding)
{
    const std::string lowered = NormalizeEncoding(encoding);
    if (lowered == "jpeg" || lowered == "jpg" || lowered == "jpe") {
        return "jpeg";
    }
    if (lowered == "png") {
        return "png";
    }
    if (lowered == "webp") {
        return "webp";
    }
    if (lowered == "avif") {
        return "avif";
    }
    return "jpeg";
}

/**
 * @brief 将未压缩的定长图像转换为 BGR Mat
 * @param image 输入的定长图像
 * @param bgr_mat 输出的 BGR Mat
 * @return 转换成功返回 true
 */
bool BuildBgrMatFromImage(const slam_common::Image& image, cv::Mat& bgr_mat)
{
    constexpr uint32_t kChannels = 3U;
    const uint32_t width = image.header.width;
    const uint32_t height = image.header.height;
    if (width == 0U || height == 0U) {
        spdlog::warn("BuildBgrMatFromImage: invalid size {}x{}", width, height);
        return false;
    }

    const std::size_t row_step = image.header.step;
    const std::size_t expected_step = static_cast<std::size_t>(width) * kChannels;
    if (row_step == 0U || row_step > std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(1, height)) {
        spdlog::warn("BuildBgrMatFromImage: invalid step {} for height {}", row_step, height);
        return false;
    }
    if (row_step < expected_step) {
        spdlog::warn("BuildBgrMatFromImage: step {} too small for width {}", row_step, width);
        return false;
    }
    const std::size_t required_size = row_step * static_cast<std::size_t>(height);
    if (required_size == 0U || required_size > slam_common::kImageMaxDataSize || required_size > image.header.payload_size) {
        spdlog::warn("BuildBgrMatFromImage: payload too small, need {}, have {}", required_size, image.header.payload_size);
        return false;
    }

    const std::string encoding = ToSafeString(image.header.encoding);
    if (encoding == "bgr8") {
        bgr_mat = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC3, const_cast<uint8_t*>(image.data.data()), row_step);
    } else if (encoding == "rgb8") {
        cv::Mat rgb(static_cast<int>(height), static_cast<int>(width), CV_8UC3, const_cast<uint8_t*>(image.data.data()), row_step);
        cv::cvtColor(rgb, bgr_mat, cv::COLOR_RGB2BGR);  // 转换为 BGR 以兼容下游编码
    } else {
        spdlog::warn("BuildBgrMatFromImage: unsupported encoding {}", encoding);
        return false;
    }

    if (!bgr_mat.isContinuous()) {
        bgr_mat = bgr_mat.clone();  // JPEG 编码需要连续内存
    }
    return true;
}

/**
 * @brief 将 BGR Mat 编码为 JPEG 数据
 * @param bgr_mat 输入的 BGR 图像
 * @param buffer 输出的压缩数据
 * @return 编码成功返回 true
 */
bool EncodeJpeg(const cv::Mat& bgr_mat, std::vector<uint8_t>& buffer)
{
    if (bgr_mat.empty() || bgr_mat.channels() != 3 || bgr_mat.type() != CV_8UC3) {
        spdlog::warn("EncodeJpeg: invalid mat type {} channels {}", bgr_mat.type(), bgr_mat.channels());
        return false;
    }
    constexpr int kJpegQuality = 85;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, kJpegQuality};
    if (!cv::imencode(".jpg", bgr_mat, buffer, params)) {
        spdlog::warn("EncodeJpeg: imencode failed for size {}x{}", bgr_mat.cols, bgr_mat.rows);
        return false;
    }
    return true;
}

/**
 * @brief 将 Mid360 点云转换为 Foxglove 点云
 */
bool ConvertMid360ToFoxglove(const slam_common::Mid360Frame& frame, FoxglovePointCloud& message)
{
    constexpr uint32_t kPointStride = 20;
    message.Clear();
    FillTimestampFromNs(frame.frame_timestamp_ns, *message.mutable_timestamp());
    message.set_frame_id(ToSafeString(frame.frame_id));
    FillIdentityPose(*message.mutable_pose());
    message.set_point_stride(kPointStride);

    if (frame.point_count == 0U) {
        return false;
    }

    message.mutable_fields()->Clear();
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

    auto* intensity_field = message.add_fields();
    intensity_field->set_name("intensity");
    intensity_field->set_offset(12);
    intensity_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* tag_field = message.add_fields();
    tag_field->set_name("tag");
    tag_field->set_offset(13);
    tag_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* line_field = message.add_fields();
    line_field->set_name("line");
    line_field->set_offset(14);
    line_field->set_type(foxglove::PackedElementField_NumericType_UINT8);

    auto* offset_field = message.add_fields();
    offset_field->set_name("offset_time");
    offset_field->set_offset(16);
    offset_field->set_type(foxglove::PackedElementField_NumericType_UINT32);

    std::string data;
    data.resize(static_cast<std::size_t>(frame.point_count) * kPointStride, 0);
    const uint64_t base_time = frame.frame_timestamp_ns;
    for (uint32_t i = 0; i < frame.point_count; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * kPointStride;
        std::memcpy(data.data() + base + 0, &frame.points[i].x, sizeof(float));
        std::memcpy(data.data() + base + 4, &frame.points[i].y, sizeof(float));
        std::memcpy(data.data() + base + 8, &frame.points[i].z, sizeof(float));
        std::memcpy(data.data() + base + 12, &frame.points[i].intensity, sizeof(uint8_t));
        std::memcpy(data.data() + base + 13, &frame.points[i].tag, sizeof(uint8_t));
        uint8_t line = 0;
        std::memcpy(data.data() + base + 14, &line, sizeof(uint8_t));
        uint32_t offset_time = 0;
        if (frame.points[i].timestamp_ns > base_time) {
            offset_time = static_cast<uint32_t>(frame.points[i].timestamp_ns - base_time);
        }
        std::memcpy(data.data() + base + 16, &offset_time, sizeof(uint32_t));
    }

    message.set_data(data);
    return true;
}

/**
 * @brief 将定长图像转换为 Foxglove 压缩图像
 */
bool ConvertImageToFoxglove(const slam_common::Image& image, FoxgloveCompressedImage& message)
{
    message.Clear();
    FillTimestampFromNs(image.header.timestamp_ns, *message.mutable_timestamp());
    message.set_frame_id(ToSafeString(image.header.frame_id));

    const std::string encoding = ToSafeString(image.header.encoding);
    if (image.header.compressed) {
        const std::size_t payload_size = std::min<std::size_t>(image.header.payload_size, slam_common::kImageMaxDataSize);
        if (payload_size == 0U) {
            spdlog::warn("ConvertImageToFoxglove: compressed image payload is empty");
            return false;
        }
        message.set_format(NormalizeImageFormat(encoding));
        message.set_data(reinterpret_cast<const char*>(image.data.data()), payload_size);
        return true;
    }

    cv::Mat bgr_mat;
    if (!BuildBgrMatFromImage(image, bgr_mat)) {
        return false;
    }

    std::vector<uint8_t> compressed;
    if (!EncodeJpeg(bgr_mat, compressed)) {
        return false;
    }

    message.set_format("jpeg");
    message.set_data(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    return true;
}

/**
 * @brief 将 IMU 转换为 Foxglove IMU
 */
bool ConvertImuToFoxglove(const slam_common::LivoxImuData& imu, FoxgloveImu& message)
{
    message.Clear();
    FillTimestampFromNs(imu.timestamp_ns, *message.mutable_timestamp());
    message.set_frame_id("");
    auto* angular_velocity = message.mutable_angular_velocity();
    angular_velocity->set_x(imu.angular_velocity[0]);
    angular_velocity->set_y(imu.angular_velocity[1]);
    angular_velocity->set_z(imu.angular_velocity[2]);

    auto* linear_acc = message.mutable_linear_acceleration();
    linear_acc->set_x(imu.linear_acceleration[0]);
    linear_acc->set_y(imu.linear_acceleration[1]);
    linear_acc->set_z(imu.linear_acceleration[2]);
    return true;
}

/**
 * @brief 将里程计数据转换为 Foxglove PoseInFrame
 */
bool ConvertOdomToFoxglove(const slam_common::OdomData& odom, FoxglovePoseInFrame& message)
{
    message.Clear();
    FillTimestampFromNs(odom.header.timestamp_ns, *message.mutable_timestamp());
    message.set_frame_id(ToSafeString(odom.header.frame_id));
    auto* pose = message.mutable_pose();
    pose->mutable_position()->set_x(odom.pose.position[0]);
    pose->mutable_position()->set_y(odom.pose.position[1]);
    pose->mutable_position()->set_z(odom.pose.position[2]);
    pose->mutable_orientation()->set_x(odom.pose.orientation[0]);
    pose->mutable_orientation()->set_y(odom.pose.orientation[1]);
    pose->mutable_orientation()->set_z(odom.pose.orientation[2]);
    pose->mutable_orientation()->set_w(odom.pose.orientation[3]);
    return true;
}

/**
 * @brief 将路径数据转换为 Foxglove PosesInFrame
 */
bool ConvertPathToFoxglove(const slam_common::PathData& path, FoxglovePosesInFrame& message)
{
    message.Clear();
    FillTimestampFromNs(path.header.timestamp_ns, *message.mutable_timestamp());
    message.set_frame_id(ToSafeString(path.header.frame_id));

    const uint32_t count = std::min<uint32_t>(path.pose_count, slam_common::kMaxPathPoses);
    for (uint32_t i = 0; i < count; ++i) {
        auto* pose = message.add_poses();
        pose->mutable_position()->set_x(path.poses[i].pose.position[0]);
        pose->mutable_position()->set_y(path.poses[i].pose.position[1]);
        pose->mutable_position()->set_z(path.poses[i].pose.position[2]);
        pose->mutable_orientation()->set_x(path.poses[i].pose.orientation[0]);
        pose->mutable_orientation()->set_y(path.poses[i].pose.orientation[1]);
        pose->mutable_orientation()->set_z(path.poses[i].pose.orientation[2]);
        pose->mutable_orientation()->set_w(path.poses[i].pose.orientation[3]);
    }
    return true;
}

/**
 * @brief 将 TF 批处理转换为 Foxglove FrameTransforms
 */
bool ConvertTransformsToFoxglove(const slam_common::FrameTransformArray& transforms, FoxgloveFrameTransforms& message)
{
    message.Clear();
    const uint32_t count = std::min<uint32_t>(transforms.transform_count, slam_common::kMaxFrameTransforms);
    for (uint32_t i = 0; i < count; ++i) {
        auto* tf_out = message.add_transforms();
        FillTimestampFromNs(transforms.transforms[i].timestamp_ns, *tf_out->mutable_timestamp());
        tf_out->set_parent_frame_id(ToSafeString(transforms.transforms[i].parent_frame_id));
        tf_out->set_child_frame_id(ToSafeString(transforms.transforms[i].child_frame_id));
        tf_out->mutable_translation()->set_x(transforms.transforms[i].transform.position[0]);
        tf_out->mutable_translation()->set_y(transforms.transforms[i].transform.position[1]);
        tf_out->mutable_translation()->set_z(transforms.transforms[i].transform.position[2]);
        tf_out->mutable_rotation()->set_x(transforms.transforms[i].transform.orientation[0]);
        tf_out->mutable_rotation()->set_y(transforms.transforms[i].transform.orientation[1]);
        tf_out->mutable_rotation()->set_z(transforms.transforms[i].transform.orientation[2]);
        tf_out->mutable_rotation()->set_w(transforms.transforms[i].transform.orientation[3]);
    }
    return true;
}
}  // namespace

/**
 * @brief 构造函数，完成 iceoryx2 初始化与 WebSocket 准备
 */
FoxgloveWebSocketBridge::FoxgloveWebSocketBridge(const Config& config) : config_(config), context_(foxglove::Context::create())
{
    spdlog::info("Initializing FoxgloveWebSocketBridge...");
    spdlog::info("  WebSocket enabled: {}", config_.websocket.enable);
    spdlog::info("  Recorder enabled: {}", config_.recorder.enable);

    auto node_result = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>();
    if (!node_result.has_value()) {
        throw std::runtime_error("Failed to create iceoryx2 node for foxglove bridge");
    }
    iox_node_ = std::make_shared<slam_common::IoxNode>(std::move(node_result.value()));

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
        slam_common::IoxPubSubConfig iox_config;

        if (topic.schema == "foxglove.PointCloud") {
            auto converter = [](const slam_common::Mid360Frame& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxglovePointCloud proto;
                if (!ConvertMid360ToFoxglove(payload, proto)) {
                    return false;
                }
                timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.timestamp());
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            pc_subs_[topic.name] = RegisterSubscriber<slam_common::Mid360Frame>(topic.name, topic.schema, converter, iox_config);
        } else if (topic.schema == "foxglove.CompressedImage") {
            auto converter = [](const slam_common::Image& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxgloveCompressedImage proto;
                if (!ConvertImageToFoxglove(payload, proto)) {
                    return false;
                }
                timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.timestamp());
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            img_subs_[topic.name] = RegisterSubscriber<slam_common::Image>(topic.name, topic.schema, converter, iox_config);
        } else if (topic.schema == "foxglove.Imu") {
            auto converter = [](const slam_common::LivoxImuData& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxgloveImu proto;
                if (!ConvertImuToFoxglove(payload, proto)) {
                    return false;
                }
                timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.timestamp());
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            iox_config.subscriber_max_buffer_size = 500;
            imu_subs_[topic.name] = RegisterSubscriber<slam_common::LivoxImuData>(topic.name, topic.schema, converter, iox_config);
        } else if (topic.schema == "foxglove.PoseInFrame") {
            auto converter = [](const slam_common::OdomData& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxglovePoseInFrame proto;
                if (!ConvertOdomToFoxglove(payload, proto)) {
                    return false;
                }
                timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.timestamp());
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            pose_subs_[topic.name] = RegisterSubscriber<slam_common::OdomData>(topic.name, topic.schema, converter, iox_config);
        } else if (topic.schema == "foxglove.PosesInFrame") {
            auto converter = [](const slam_common::PathData& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxglovePosesInFrame proto;
                if (!ConvertPathToFoxglove(payload, proto)) {
                    return false;
                }
                timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.timestamp());
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            poses_subs_[topic.name] = RegisterSubscriber<slam_common::PathData>(topic.name, topic.schema, converter, iox_config);
        } else if (topic.schema == "foxglove.FrameTransforms") {
            auto converter = [](const slam_common::FrameTransformArray& payload, std::string& buffer, uint64_t& timestamp_ns) {
                FoxgloveFrameTransforms proto;
                if (!ConvertTransformsToFoxglove(payload, proto)) {
                    return false;
                }
                if (proto.transforms_size() > 0) {
                    timestamp_ns = FoxgloveWebSocketBridge::ToNanoseconds(proto.transforms(0).timestamp());
                } else {
                    timestamp_ns = 0U;
                }
                return FoxgloveWebSocketBridge::SerializeMessage(proto, buffer);
            };
            frame_tf_subs_[topic.name] = RegisterSubscriber<slam_common::FrameTransformArray>(topic.name, topic.schema, converter, iox_config);
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
            // 先从 iceoryx2 拉取数据触发回调填充缓存
            auto poll_subscribers = [](auto& subs_map) {
                for (auto& kv : subs_map) {
                    if (!kv.second || !kv.second->IsReady()) {
                        continue;
                    }
                    kv.second->ReceiveAll();
                }
            };
            poll_subscribers(pc_subs_);
            poll_subscribers(img_subs_);
            poll_subscribers(imu_subs_);
            poll_subscribers(pose_subs_);
            poll_subscribers(poses_subs_);
            poll_subscribers(frame_tf_subs_);

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
std::shared_ptr<slam_common::IoxSubscriber<MessageType>> FoxgloveWebSocketBridge::RegisterSubscriber(
    const std::string& topic_name,
    const std::string& schema_name,
    const std::function<bool(const MessageType&, std::string&, uint64_t&)>& converter,
    const slam_common::IoxPubSubConfig& iox_config)
{
    if (!iox_node_) {
        throw std::runtime_error("iceoryx2 node is not initialized");
    }

    auto subscriber = std::make_shared<slam_common::IoxSubscriber<MessageType>>(
        iox_node_,
        topic_name,
        [this, topic_name, schema_name, converter](const MessageType& payload) {
            std::string buffer;
            uint64_t timestamp_ns = 0;
            if (!converter || !converter(payload, buffer, timestamp_ns)) {
                error_count_.at(topic_name)++;
                spdlog::warn("Converter failed for topic {}", topic_name);
                return;
            }

            const uint64_t aligned_ts = EnsureGlobalMonotonic(AlignTimestamp(timestamp_ns));
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_packets_[topic_name].push_back({std::move(buffer), aligned_ts});
            }
        }, iox_config);

    spdlog::info("iceoryx2 subscriber created for topic {} ({})", topic_name, schema_name);
    return subscriber;
}

// 显式实例化
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::Mid360Frame>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::Mid360Frame>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::Mid360Frame&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::Image>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::Image>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::Image&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::LivoxImuData>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::LivoxImuData>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::LivoxImuData&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::OdomData>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::OdomData>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::OdomData&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::PathData>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::PathData>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::PathData&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
template std::shared_ptr<slam_common::IoxSubscriber<slam_common::FrameTransformArray>>
FoxgloveWebSocketBridge::RegisterSubscriber<slam_common::FrameTransformArray>(
    const std::string&,
    const std::string&,
    const std::function<bool(const slam_common::FrameTransformArray&, std::string&, uint64_t&)>&,
    const slam_common::IoxPubSubConfig&);
}  // namespace ms_slam::slam_recorder
