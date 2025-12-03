#include "slam_recorder/bag_tool.hpp"

#include <csignal>
#include <poll.h>
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
#include <utility>
#include <vector>

#include <cstring>
#include <mcap/reader.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <iox2/iceoryx2.hpp>
#include <slam_common/crash_logger.hpp>
#include <slam_common/iceoryx_pub_sub.hpp>
#include <slam_common/sensor_struct.hpp>

#include "slam_recorder/ros1_msg.hpp"

namespace ms_slam::slam_recorder
{

namespace
{

using ms_slam::slam_common::FrameTransformArray;
using ms_slam::slam_common::Image;
using ms_slam::slam_common::kMaxFrameTransforms;
using ms_slam::slam_common::kMaxPathPoses;
using ms_slam::slam_common::kMid360MaxPoints;
using ms_slam::slam_common::LivoxImuData;
using ms_slam::slam_common::Mid360Frame;
using ms_slam::slam_common::OdomData;
using ms_slam::slam_common::PathData;
using ms_slam::slam_common::TimeFrameHeader;

/**
 * @brief 内部使用的消息类型分类
 */
enum class MessageKind { PointCloud, Image, CompressedImage, Imu, PoseInFrame, PosesInFrame, FrameTransforms, Unsupported };

/**
 * @brief 消息描述信息，用于快速判断数据来源类型
 */
struct MessageDescriptor {
    MessageKind kind{MessageKind::Unsupported};
    bool is_ros{false};
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

    auto set_ros = [&](MessageKind kind) {
        descriptor.kind = kind;
        descriptor.is_ros = true;
    };

    if (schema_name == "sensor_msgs/PointCloud2") {
        set_ros(MessageKind::PointCloud);
        return descriptor;
    }
    if (schema_name == "sensor_msgs/Image") {
        set_ros(MessageKind::Image);
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
    resolved.publish_service = topic;
    resolved.schema = "";

    const auto it = config.topics.find(topic);
    if (it != config.topics.end()) {
        if (!it->second.publish_service.empty()) {
            resolved.publish_service = it->second.publish_service;
        }
        resolved.playback = config.playback_enabled && it->second.playback;
        resolved.schema = it->second.schema;
    }

    return resolved;
}

/**
 * @brief 回放上下文，包含调度、节点及 iceoryx2 发布器缓存
 */
struct PlaybackContext {
    bool enabled{false};
    bool sync_time{true};
    double rate{1.0};
    slam_common::IoxPubSubConfig pubsub_config{};
    std::shared_ptr<slam_common::IoxNode> node;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<Mid360Frame>>> pointcloud_publishers;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<Image>>> image_publishers;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<LivoxImuData>>> imu_publishers;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<OdomData>>> odom_publishers;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<PathData>>> path_publishers;
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<FrameTransformArray>>> tf_publishers;
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
    size_t image{0};
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
 * @brief 播放窗口判定结果
 */
enum class WindowDecision { Process, Skip, Stop };

/**
 * @brief 播放窗口管理，负责起止时间判定
 */
struct PlaybackWindow {
    std::optional<uint64_t> bag_start_ns;   ///< 记录文件首帧时间戳
    std::optional<uint64_t> process_start;  ///< 窗口起始时间戳
    std::optional<uint64_t> process_end;    ///< 窗口结束时间戳
    uint64_t start_offset_ns{0};            ///< 起始偏移
    uint64_t duration_ns{0};                ///< 播放时长（纳秒）
    bool has_duration{false};               ///< 是否限定时长

    /**
     * @brief 判定当前时间戳的处理策略
     * @param timestamp_ns 当前消息时间戳
     * @param stats 消息统计信息（用于窗口跳过计数）
     * @return 窗口判定结果
     */
    WindowDecision Evaluate(uint64_t timestamp_ns, MessageStats& stats)
    {
        if (!bag_start_ns.has_value()) {
            bag_start_ns = timestamp_ns;
            process_start = bag_start_ns.value() + start_offset_ns;
            if (has_duration) {
                process_end = process_start.value() + duration_ns;
            }
        }

        const uint64_t start_ns = process_start.value_or(timestamp_ns);
        if (timestamp_ns < start_ns) {
            ++stats.window_skipped;
            return WindowDecision::Skip;
        }
        if (has_duration && process_end.has_value() && timestamp_ns > process_end.value()) {
            return WindowDecision::Stop;
        }
        return WindowDecision::Process;
    }
};

std::atomic<bool> g_interrupted{false};  ///< SIGINT 退出标志

/**
 * @brief 获取或创建指定类型的 iceoryx2 发布器
 * @tparam Payload 负载类型
 * @param cache 发布器缓存
 * @param ctx 回放上下文
 * @param service_name 服务名称
 * @return 发布器指针，失败返回 nullptr
 */
template <typename Payload>
slam_common::IoxPublisher<Payload>* EnsureIoxPublisher(
    std::unordered_map<std::string, std::unique_ptr<slam_common::IoxPublisher<Payload>>>& cache,
    PlaybackContext& ctx,
    const std::string& service_name)
{
    auto it = cache.find(service_name);
    if (it != cache.end()) {
        return it->second.get();
    }
    if (!ctx.node) {
        spdlog::error("Iox node not initialized, cannot create publisher for {}", service_name);
        return nullptr;
    }

    auto publisher =
        std::make_unique<slam_common::IoxPublisher<Payload>>(ctx.node, service_name, nullptr, ctx.pubsub_config);
    auto* raw_ptr = publisher.get();
    cache.emplace(service_name, std::move(publisher));
    spdlog::info("Iox publisher created for {}", service_name);
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
 * @brief 将 std::string 安全复制到定长数组
 * @tparam N 目标数组长度
 * @param src 源字符串
 * @param dst 目标数组
 */
template <std::size_t N>
void CopyStringToArray(const std::string& src, std::array<char, N>& dst)
{
    dst.fill(0);
    const std::size_t length = std::min<std::size_t>(N - 1, src.size());
    std::memcpy(dst.data(), src.data(), length);
}

/**
 * @brief 为已有对象创建共享指针别名（不接管所有权）
 * @tparam Payload 消息类型
 * @param ptr 原始指针
 * @return 指向原始对象的共享指针
 * @note 删除器为空操作，调用方需保证原始指针生命周期
 */
template <typename Payload>
std::shared_ptr<Payload> MakeSharedAlias(Payload* ptr)
{
    return std::shared_ptr<Payload>(ptr, [](Payload*) {});
}

/**
 * @brief 将秒与纳秒字段转换为统一的纳秒时间戳
 * @param sec 秒
 * @param nsec 纳秒
 * @return 转换后的纳秒时间戳，非法输入返回 0
 */
inline uint64_t ToNanoseconds(uint32_t sec, uint32_t nsec)
{
    constexpr uint64_t kNsPerSec = 1'000'000'000ULL;
    if (nsec >= kNsPerSec) {
        return 0U;
    }
    return static_cast<uint64_t>(sec) * kNsPerSec + static_cast<uint64_t>(nsec);
}

/**
 * @brief 使用 BGR 数据填充 Image 结构
 * @param timestamp_ns 时间戳（纳秒）
 * @param frame_id 坐标系 ID
 * @param width 图像宽度
 * @param height 图像高度
 * @param payload 输入的 BGR 数据缓冲区
 * @param payload_size 缓冲区长度
 * @param image 输出的图像结构
 * @return 成功填充返回 true
 */
bool FillBgrImage(
    uint64_t timestamp_ns, const std::string& frame_id, uint32_t width, uint32_t height, const uint8_t* payload, std::size_t payload_size, Image& image)
{
    constexpr std::size_t kChannels = 3U;
    const std::size_t width_sz = static_cast<std::size_t>(width);
    const std::size_t height_sz = static_cast<std::size_t>(height);
    if (width_sz == 0U || height_sz == 0U) {
        spdlog::warn("FillBgrImage: invalid image size {}x{}", width, height);
        return false;
    }
    if (width_sz > std::numeric_limits<std::size_t>::max() / height_sz || width_sz * height_sz > std::numeric_limits<std::size_t>::max() / kChannels) {
        spdlog::warn("FillBgrImage: image size overflow {}x{}", width, height);
        return false;
    }

    const std::size_t expected_size = width_sz * height_sz * kChannels;
    if (expected_size > slam_common::kImageMaxDataSize) {
        spdlog::warn("FillBgrImage: payload {} exceeds buffer {}", expected_size, slam_common::kImageMaxDataSize);
        return false;
    }
    if (payload == nullptr || payload_size < expected_size) {
        spdlog::warn("FillBgrImage: payload missing or too small, have {}, need {}", payload_size, expected_size);
        return false;
    }
    if (expected_size > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        spdlog::warn("FillBgrImage: payload size overflow {}", expected_size);
        return false;
    }

    image.header.timestamp_ns = timestamp_ns;
    CopyStringToArray(frame_id, image.header.frame_id);
    CopyStringToArray("bgr8", image.header.encoding);
    image.header.width = width;
    image.header.height = height;
    image.header.step = static_cast<uint32_t>(width_sz * kChannels);
    image.header.payload_size = static_cast<uint32_t>(expected_size);
    image.header.compressed = false;

    std::memcpy(image.data.data(), payload, expected_size);
    return true;
}

bool ConvertPointCloud(const mcap::MessageView& view, const MessageDescriptor& descriptor, Mid360Frame& frame)
{
    if (!descriptor.is_ros) {
        spdlog::warn("Non-ROS pointcloud unsupported");
        return false;
    }
    if (descriptor.is_livox) {
        ROS1LivoxCustomMsg livox_msg;
        if (!livox_msg.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
            const std::string topic = view.channel ? view.channel->topic : "<unknown>";
            spdlog::error("Failed to parse Livox CustomMsg at topic {}", topic);
            return false;
        }
        if (livox_msg.points.empty()) {
            spdlog::warn("Empty Livox pointcloud, skip");
            return false;
        }
        frame.index = livox_msg.header.seq;
        frame.frame_timestamp_ns = ToNanoseconds(livox_msg.header.stamp_sec, livox_msg.header.stamp_nsec);
        CopyStringToArray(livox_msg.header.frame_id, frame.frame_id);
        frame.point_count = static_cast<uint32_t>(std::min<std::size_t>(livox_msg.points.size(), kMid360MaxPoints));
        for (uint32_t i = 0; i < frame.point_count; ++i) {
            const auto& src = livox_msg.points[i];
            auto& dst = frame.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.reflectivity;
            dst.tag = src.tag;
            dst.timestamp_ns = frame.frame_timestamp_ns + static_cast<uint64_t>(src.offset_time);
        }
        return true;
    }

    ROS1PointCloud2 pc2;
    if (!pc2.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/PointCloud2 at topic {}", topic);
        return false;
    }

    int32_t x_offset = -1;
    int32_t y_offset = -1;
    int32_t z_offset = -1;
    int32_t intensity_offset = -1;
    int32_t tag_offset = -1;
    int32_t offset_time_offset = -1;
    for (const auto& field : pc2.fields) {
        if (field.name == "x")
            x_offset = static_cast<int32_t>(field.offset);
        else if (field.name == "y")
            y_offset = static_cast<int32_t>(field.offset);
        else if (field.name == "z")
            z_offset = static_cast<int32_t>(field.offset);
        else if (field.name == "intensity")
            intensity_offset = static_cast<int32_t>(field.offset);
        else if (field.name == "tag" || field.name == "line")
            tag_offset = static_cast<int32_t>(field.offset);
        else if (field.name == "offset_time")
            offset_time_offset = static_cast<int32_t>(field.offset);
    }
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        spdlog::warn("Pointcloud missing xyz, skip");
        return false;
    }

    const std::size_t stride = pc2.point_step;
    const auto offset_in_range = [stride](int32_t off, std::size_t width) { return off >= 0 && static_cast<std::size_t>(off) + width <= stride; };
    if (!offset_in_range(x_offset, sizeof(float)) || !offset_in_range(y_offset, sizeof(float)) || !offset_in_range(z_offset, sizeof(float))) {
        spdlog::warn("Pointcloud offsets exceed stride {}", stride);
        return false;
    }

    const std::size_t total_points = pc2.data.size() / stride;
    if (total_points == 0) {
        spdlog::warn("Empty pointcloud, skip");
        return false;
    }

    frame.index = pc2.header.seq;
    frame.frame_timestamp_ns = ToNanoseconds(pc2.header.stamp_sec, pc2.header.stamp_nsec);
    CopyStringToArray(pc2.header.frame_id, frame.frame_id);
    frame.point_count = static_cast<uint32_t>(std::min<std::size_t>(total_points, kMid360MaxPoints));

    for (uint32_t i = 0; i < frame.point_count; ++i) {
        const auto* base_ptr = pc2.data.data() + i * stride;
        auto& dst = frame.points[i];
        std::memcpy(&dst.x, base_ptr + x_offset, sizeof(float));
        std::memcpy(&dst.y, base_ptr + y_offset, sizeof(float));
        std::memcpy(&dst.z, base_ptr + z_offset, sizeof(float));
        if (intensity_offset >= 0) {
            std::memcpy(&dst.intensity, base_ptr + intensity_offset, sizeof(uint8_t));
        }
        if (tag_offset >= 0) {
            std::memcpy(&dst.tag, base_ptr + tag_offset, sizeof(uint8_t));
        }
        if (offset_time_offset >= 0) {
            uint32_t offset_time = 0;
            std::memcpy(&offset_time, base_ptr + offset_time_offset, sizeof(uint32_t));
            dst.timestamp_ns = frame.frame_timestamp_ns + static_cast<uint64_t>(offset_time);
        } else {
            dst.timestamp_ns = frame.frame_timestamp_ns;
        }
    }
    return true;
}

/**
 * @brief 解析并解码压缩图像
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param image 输出图像
 * @return 成功返回 true
 */
bool ConvertCompressedImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, Image& image)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1CompressedImage ros_img;
    if (!ros_img.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/CompressedImage at topic {}", topic);
        return false;
    }
    if (ros_img.data.empty()) {
        spdlog::warn("ConvertCompressedImage: empty payload");
        return false;
    }

    cv::Mat decoded = cv::imdecode(ros_img.data, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        spdlog::warn("ConvertCompressedImage: imdecode failed");
        return false;
    }
    if (decoded.channels() != 3 || decoded.type() != CV_8UC3) {
        spdlog::warn("ConvertCompressedImage: unsupported decoded format type={}, channels={}", decoded.type(), decoded.channels());
        return false;
    }
    if (!decoded.isContinuous()) {
        decoded = decoded.clone();  // 确保内存连续便于 memcpy
    }

    if (decoded.cols <= 0 || decoded.rows <= 0) {
        spdlog::warn("ConvertCompressedImage: decoded image size invalid {}x{}", decoded.cols, decoded.rows);
        return false;
    }
    const uint32_t width = static_cast<uint32_t>(decoded.cols);
    const uint32_t height = static_cast<uint32_t>(decoded.rows);
    const std::size_t payload_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    const std::size_t available_size = static_cast<std::size_t>(decoded.total()) * decoded.elemSize();
    if (available_size < payload_size) {
        spdlog::warn("ConvertCompressedImage: decoded buffer smaller than expected, have {}, need {}", available_size, payload_size);
        return false;
    }
    return FillBgrImage(
        ToNanoseconds(ros_img.header.stamp_sec, ros_img.header.stamp_nsec), ros_img.header.frame_id, width, height, decoded.data, payload_size, image);
}

/**
 * @brief 解析原始图像消息
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param image 输出图像
 * @return 成功返回 true
 */
bool ConvertRawImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, Image& image)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1Image ros_img;
    if (!ros_img.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/Image at topic {}", topic);
        return false;
    }

    const uint32_t expected_step = ros_img.width * 3U;
    const std::size_t expected_size = static_cast<std::size_t>(expected_step) * static_cast<std::size_t>(ros_img.height);
    if (ros_img.width == 0U || ros_img.height == 0U) {
        spdlog::warn("ConvertRawImage: invalid image size {}x{}", ros_img.width, ros_img.height);
        return false;
    }
    if (ros_img.step != expected_step) {
        spdlog::warn("ConvertRawImage: unsupported step {} for width {}", ros_img.step, ros_img.width);
        return false;
    }
    if (expected_size > slam_common::kImageMaxDataSize) {
        spdlog::warn("ConvertRawImage: payload {} exceeds buffer {}", expected_size, slam_common::kImageMaxDataSize);
        return false;
    }
    if (ros_img.data.size() < expected_size) {
        spdlog::warn("ConvertRawImage: payload too small, have {}, need {}", ros_img.data.size(), expected_size);
        return false;
    }

    const uint64_t timestamp_ns = ToNanoseconds(ros_img.header.stamp_sec, ros_img.header.stamp_nsec);
    if (ros_img.encoding == "bgr8") {
        return FillBgrImage(timestamp_ns, ros_img.header.frame_id, ros_img.width, ros_img.height, ros_img.data.data(), expected_size, image);
    }
    if (ros_img.encoding == "rgb8") {
        std::vector<uint8_t> bgr(expected_size);
        for (std::size_t i = 0; i + 2 < expected_size; i += 3) {
            // 将 RGB 转换为 BGR
            bgr[i] = ros_img.data[i + 2];
            bgr[i + 1] = ros_img.data[i + 1];
            bgr[i + 2] = ros_img.data[i];
        }
        return FillBgrImage(timestamp_ns, ros_img.header.frame_id, ros_img.width, ros_img.height, bgr.data(), expected_size, image);
    }

    const std::string topic = view.channel ? view.channel->topic : "<unknown>";
    spdlog::warn("ConvertRawImage: unsupported encoding {} at topic {}", ros_img.encoding, topic);
    return false;
}

bool ConvertImu(const mcap::MessageView& view, const MessageDescriptor& descriptor, LivoxImuData& imu)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1Imu ros_imu;
    if (!ros_imu.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/Imu at topic {}", topic);
        return false;
    }
    imu.timestamp_ns = static_cast<uint64_t>(ros_imu.header.stamp_sec) * 1'000'000'000ULL + ros_imu.header.stamp_nsec;
    imu.index = ros_imu.header.seq;
    imu.angular_velocity = {ros_imu.angular_velocity.x, ros_imu.angular_velocity.y, ros_imu.angular_velocity.z};
    imu.linear_acceleration = {ros_imu.linear_acceleration.x, ros_imu.linear_acceleration.y, ros_imu.linear_acceleration.z};
    return true;
}

bool ConvertPath(const mcap::MessageView& view, const MessageDescriptor& descriptor, PathData& path)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1Path ros_path;
    if (!ros_path.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse nav_msgs/Path at topic {}", topic);
        return false;
    }
    path.header.timestamp_ns = static_cast<uint64_t>(ros_path.header.stamp_sec) * 1'000'000'000ULL + ros_path.header.stamp_nsec;
    CopyStringToArray(ros_path.header.frame_id, path.header.frame_id);
    const std::size_t count = std::min<std::size_t>(ros_path.poses.size(), kMaxPathPoses);
    path.pose_count = static_cast<uint32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& pose = ros_path.poses[i].pose;
        path.poses[i].timestamp_ns = path.header.timestamp_ns;
        path.poses[i].pose.position = {pose.position.x, pose.position.y, pose.position.z};
        path.poses[i].pose.orientation = {pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
    }
    return true;
}

bool ConvertOdometry(const mcap::MessageView& view, const MessageDescriptor& descriptor, OdomData& odom)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1Odometry ros_odom;
    if (!ros_odom.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse nav_msgs/Odometry at topic {}", topic);
        return false;
    }
    odom.header.timestamp_ns = static_cast<uint64_t>(ros_odom.header.stamp_sec) * 1'000'000'000ULL + ros_odom.header.stamp_nsec;
    CopyStringToArray(ros_odom.header.frame_id, odom.header.frame_id);
    odom.pose.position = {ros_odom.pose.position.x, ros_odom.pose.position.y, ros_odom.pose.position.z};
    odom.pose.orientation = {ros_odom.pose.orientation.x, ros_odom.pose.orientation.y, ros_odom.pose.orientation.z, ros_odom.pose.orientation.w};
    odom.pose_covariance.fill(0.0);
    odom.linear_velocity = {ros_odom.twist.linear.x, ros_odom.twist.linear.y, ros_odom.twist.linear.z};
    odom.angular_velocity = {ros_odom.twist.angular.x, ros_odom.twist.angular.y, ros_odom.twist.angular.z};
    odom.twist_covariance.fill(0.0);
    odom.child_frame_id.fill(0);
    return true;
}

bool ConvertTfMessage(const mcap::MessageView& view, const MessageDescriptor& descriptor, FrameTransformArray& transforms)
{
    if (!descriptor.is_ros) {
        return false;
    }
    ROS1TFMessage tf;
    if (!tf.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse tf2_msgs/TFMessage at topic {}", topic);
        return false;
    }
    const std::size_t count = std::min<std::size_t>(tf.transforms.size(), kMaxFrameTransforms);
    transforms.transform_count = static_cast<uint32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& src = tf.transforms[i];
        auto& dst = transforms.transforms[i];
        dst.timestamp_ns = static_cast<uint64_t>(src.header.stamp_sec) * 1'000'000'000ULL + src.header.stamp_nsec;
        CopyStringToArray(src.header.frame_id, dst.parent_frame_id);
        CopyStringToArray(src.child_frame_id, dst.child_frame_id);
        dst.transform.position = {src.transform.translation.x, src.transform.translation.y, src.transform.translation.z};
        dst.transform.orientation = {src.transform.rotation.x, src.transform.rotation.y, src.transform.rotation.z, src.transform.rotation.w};
    }
    return true;
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

    auto node_result = iox2::NodeBuilder().create<iox2::ServiceType::Ipc>();
    if (!node_result.has_value()) {
        spdlog::error("Failed to create iceoryx2 node for playback, publishing disabled");
        ctx.enabled = false;
        return ctx;
    }
    ctx.node = std::make_shared<slam_common::IoxNode>(std::move(node_result.value()));

    spdlog::info("Playback enabled via iceoryx2 (rate {:.3f}, sync_time={})", ctx.rate, ctx.sync_time);
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
            if (g_interrupted.load(std::memory_order_relaxed)) {
                break;
            }
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
 * @brief 处理暂停与中断逻辑，统一维护播放起始时间
 * @param keyboard 键盘控制句柄
 * @param playback 回放上下文
 * @param pause_started 暂停开始时间（用于恢复墙钟基准）
 * @return 继续处理返回 true，需退出返回 false
 */
bool HandlePauseAndInterrupt(
    KeyboardControl& keyboard,
    PlaybackContext& playback,
    std::optional<std::chrono::steady_clock::time_point>& pause_started)
{
    if (keyboard.paused.load()) {
        if (!pause_started.has_value()) {
            pause_started = std::chrono::steady_clock::now();
        }
    }
    while (keyboard.paused.load() && !keyboard.stop.load()) {
        if (g_interrupted.load(std::memory_order_relaxed)) {
            keyboard.stop.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (keyboard.stop.load()) {
        return false;
    }
    if (pause_started.has_value()) {
        if (playback.first_timestamp.has_value()) {
            playback.start_wall_time += std::chrono::steady_clock::now() - *pause_started;
        }
        pause_started.reset();
    }
    return true;
}

// 发布逻辑改为在主循环内按类型直接发布

/**
 * @brief 运行期上下文，封装录制与回放状态
 */
struct ToolRuntime {
    ToolConfig config;
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
        spdlog::info("Failed to read MCAP summary: {}", summary_status.message);
    } else if (const auto stats_opt = reader.statistics(); stats_opt.has_value()) {
        total_messages = stats_opt->messageCount;
    }
    if (total_messages > 0) {
        spdlog::info("Estimated messages: {}", total_messages);
    } else {
        spdlog::info("Estimated messages: unknown");
    }

    ToolRuntime runtime{config, CreatePlaybackContext(config)};
    MessageStats stats;
    KeyboardControl keyboard;
    StartKeyboardControl(keyboard);
    if (keyboard.paused.load()) {
        spdlog::info("Playback paused. Press space to start");
    }

    auto messages = reader.readMessages();
    uint64_t processed = 0;
    uint64_t next_progress = total_messages > 0 ? std::max<uint64_t>(1, total_messages / 20) : 1000;
    std::optional<std::chrono::steady_clock::time_point> pause_started;
    PlaybackWindow window;
    window.start_offset_ns = start_offset_ns;
    window.duration_ns = duration_ns;
    window.has_duration = has_duration;
    for (auto it = messages.begin(); it != messages.end(); ++it) {
        const auto& view = *it;
        const std::string topic = view.channel ? view.channel->topic : "";
        if (topic.empty()) {
            spdlog::warn("Encountered empty topic, skip message");
            ++stats.skipped;
            continue;
        }

        const uint64_t timestamp_ns = view.message.logTime != 0 ? view.message.logTime : view.message.publishTime;
        if (g_interrupted.load(std::memory_order_relaxed)) {
            spdlog::warn("SIGINT received, stopping gracefully...");
            break;
        }

        const WindowDecision window_decision = window.Evaluate(timestamp_ns, stats);
        if (window_decision == WindowDecision::Skip) {
            continue;
        }
        if (window_decision == WindowDecision::Stop) {
            spdlog::info("Processing window reached end timestamp, stopping");
            break;
        }

        const MessageDescriptor descriptor = DescribeMessage(view);
        if (descriptor.kind == MessageKind::Unsupported) {
            ++stats.skipped;
            continue;
        }

        const TopicSettings topic_settings = ResolveTopic(config, topic);
        if (!topic_settings.playback) {
            ++stats.filtered;
            continue;
        }

        if (!HandlePauseAndInterrupt(keyboard, runtime.playback, pause_started)) {
            break;
        }

        switch (descriptor.kind) {
            case MessageKind::PointCloud: {
                auto* pub = EnsureIoxPublisher(runtime.playback.pointcloud_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](Mid360Frame& payload) {
                        if (!ConvertPointCloud(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.pointcloud;
                    }
                }
                break;
            }
            case MessageKind::Image: {
                auto* pub = EnsureIoxPublisher(runtime.playback.image_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](Image& payload) {
                        if (!ConvertRawImage(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.image;
                    }
                }
                break;
            }
            case MessageKind::CompressedImage: {
                auto* pub = EnsureIoxPublisher(runtime.playback.image_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](Image& payload) {
                        if (!ConvertCompressedImage(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.compressed_image;
                    }
                }
                break;
            }
            case MessageKind::Imu: {
                runtime.playback.pubsub_config.subscriber_max_buffer_size = 500;
                auto* pub = EnsureIoxPublisher(runtime.playback.imu_publishers, runtime.playback, topic_settings.publish_service);
                runtime.playback.pubsub_config.subscriber_max_buffer_size = 10;
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](LivoxImuData& payload) {
                        if (!ConvertImu(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.imu;
                    }
                }
                break;
            }
            case MessageKind::PoseInFrame: {
                auto* pub = EnsureIoxPublisher(runtime.playback.odom_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](OdomData& payload) {
                        if (!ConvertOdometry(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.pose;
                    }
                }
                break;
            }
            case MessageKind::PosesInFrame: {
                auto* pub = EnsureIoxPublisher(runtime.playback.path_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](PathData& payload) {
                        if (!ConvertPath(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.path;
                    }
                }
                break;
            }
            case MessageKind::FrameTransforms: {
                auto* pub = EnsureIoxPublisher(runtime.playback.tf_publishers, runtime.playback, topic_settings.publish_service);
                if (pub != nullptr) {
                    const bool sent = pub->PublishWithBuilder([&](FrameTransformArray& payload) {
                        if (!ConvertTfMessage(view, descriptor, payload)) {
                            return false;
                        }
                        PlaybackSleep(runtime.playback, timestamp_ns);
                        return true;
                    });
                    if (sent) {
                        ++stats.tf;
                    }
                }
                break;
            }
            case MessageKind::Unsupported:
            default:
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
    StopKeyboardControl(keyboard);

    spdlog::info("PointCloud messages: {}", stats.pointcloud);
    spdlog::info("Image messages: {}", stats.image);
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
        config.playback_enabled = playback_node["enable"].as<bool>(false);
        config.playback_rate = playback_node["rate"].as<double>(1.0);
        config.playback_sync_time = playback_node["synchronize_time"].as<bool>(true);
    }

    const auto window_node = bag_node["time_window"];
    if (window_node) {
        config.processing_start_seconds = window_node["start_seconds"].as<double>(config.processing_start_seconds);
        config.processing_duration_seconds = window_node["duration_seconds"].as<double>(config.processing_duration_seconds);
    }

    const auto topics_node = bag_node["topics"];
    if (topics_node) {
        // 提供全局默认值后，再读取具体的话题设置
        config.default_playback = topics_node["default_playback_enabled"].as<bool>(true);
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
                if (entry["schema"]) {
                    settings.schema = entry["schema"].as<std::string>();
                }
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
        ms_slam::slam_common::LoggerConfig log_config;
        log_config.log_file_path = "bag_tool.log";
        auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
        dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_config.log_file_path, true));
        auto logger = std::make_shared<spdlog::logger>("crash_logger", dup_filter);

        logger->set_pattern(log_config.log_pattern);
        logger->flush_on(spdlog::level::warn);

        spdlog::set_default_logger(logger);

        if (!SLAM_CRASH_LOGGER_INIT(logger)) {
            spdlog::error("Failed to initialize crash logger!");
            return 1;
        }
        spdlog::info("Crash logger initialized successfully");

        auto config = ms_slam::slam_recorder::LoadBagToolConfig(config_path);
        return ms_slam::slam_recorder::RunTool(config);
    } catch (const std::exception& ex) {
        spdlog::critical("bag_tool failed: {}", ex.what());
        return 1;
    }
}
