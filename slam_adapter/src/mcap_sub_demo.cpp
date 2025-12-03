#include <csignal>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <easy/arbitrary_value.h>
#include <easy/profiler.h>
#include <mcap/reader.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <slam_common/crash_logger.hpp>
#include <slam_common/iceoryx_pub_sub.hpp>
#include <slam_common/sensor_struct.hpp>
#include <slam_recorder/ros1_msg.hpp>
#include <slam_core/odometry.hpp>
#include <yaml-cpp/yaml.h>

#include "slam_adapter/config_loader.hpp"
#include "slam_adapter/sensor_preprocess.hpp"
#include "slam_adapter/sensor_publish.hpp"

using namespace ms_slam::slam_common;
using namespace ms_slam::slam_core;
using namespace ms_slam::slam_adapter;

namespace
{

std::atomic<bool> shouldExit{false};
constexpr char kProfileDumpPath[] = "/home/ubuntu/data/mcap_profile.prof";

/**
 * @brief 处理SIGINT信号并设置退出标记
 * @param signal 捕获到的信号编号
 */
void HandleSignal(int signal)
{
    (void)signal;
    shouldExit.store(true);
}

/**
 * @brief MCAP 消息类别
 */
enum class MessageKind { PointCloud, CompressedImage, Image, Imu, Unsupported };

/**
 * @brief 消息特征描述
 */
struct MessageDescriptor {
    MessageKind kind{MessageKind::Unsupported};  ///< 消息类型
    bool is_livox{false};                        ///< 是否为 Livox 自定义点云
};

/**
 * @brief 将 MCAP 消息的元信息转换为内部描述
 * @param view MCAP 消息视图
 * @return 消息描述
 */
MessageDescriptor DescribeMessage(const mcap::MessageView& view)
{
    MessageDescriptor descriptor;
    const std::string schema_name = view.schema ? view.schema->name : "";

    if (schema_name == "sensor_msgs/PointCloud2") {
        descriptor.kind = MessageKind::PointCloud;
        return descriptor;
    }
    if (schema_name == "sensor_msgs/CompressedImage") {
        descriptor.kind = MessageKind::CompressedImage;
        return descriptor;
    }
    if (schema_name == "sensor_msgs/Image") {
        descriptor.kind = MessageKind::Image;
        return descriptor;
    }
    if (schema_name == "sensor_msgs/Imu") {
        descriptor.kind = MessageKind::Imu;
        return descriptor;
    }
    if (schema_name == "livox_ros_driver2/CustomMsg" || schema_name == "livox_ros_driver/CustomMsg") {
        descriptor.kind = MessageKind::PointCloud;
        descriptor.is_livox = true;
        return descriptor;
    }

    descriptor.kind = MessageKind::Unsupported;
    return descriptor;
}

/**
 * @brief 将 Livox 自定义点云转为 Mid360Frame
 * @param view MCAP 消息视图
 * @param frame 输出点云帧
 * @return 转换成功返回 true
 */
bool ConvertLivoxPointCloud(const mcap::MessageView& view, Mid360Frame& frame)
{
    ROS1LivoxCustomMsg livox_msg;
    if (!livox_msg.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse Livox CustomMsg at topic {}", topic);
        return false;
    }

    frame.index = livox_msg.header.seq;
    frame.frame_timestamp_ns = static_cast<uint64_t>(livox_msg.header.stamp_sec) * 1000000000ULL + livox_msg.header.stamp_nsec;
    frame.frame_id.fill('\0');
    std::memcpy(frame.frame_id.data(), livox_msg.header.frame_id.data(),
                std::min(livox_msg.header.frame_id.size(), frame.frame_id.size() - 1));
    const uint32_t count = std::min<uint32_t>(livox_msg.point_num, kMid360MaxPoints);
    frame.point_count = count;

    for (uint32_t i = 0; i < count; ++i) {
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

/**
 * @brief 解析点云消息，目前仅支持 Livox 自定义格式
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param frame 输出点云帧
 * @return 成功返回 true
 */
bool ConvertPointCloud(const mcap::MessageView& view, const MessageDescriptor& descriptor, Mid360Frame& frame)
{
    if (!descriptor.is_livox) {
        spdlog::warn("ConvertPointCloud: unsupported schema {}", view.schema ? view.schema->name : "<none>");
        return false;
    }
    return ConvertLivoxPointCloud(view, frame);
}

/**
 * @brief 解析压缩图像并转换为定长图像结构
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param image_msg 输出图像
 * @return 成功返回 true
 */
bool ConvertCompressedImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, ms_slam::slam_common::Image& image_msg)
{
    if (descriptor.kind != MessageKind::CompressedImage) {
        return false;
    }

    ROS1CompressedImage ros_img;
    if (!ros_img.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/CompressedImage at topic {}", topic);
        return false;
    }

    std::vector<std::uint8_t> buffer(ros_img.data.begin(), ros_img.data.end());
    const auto decoded_mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (decoded_mat.empty()) {
        spdlog::warn("ConvertCompressedImage: failed to decode compressed image");
        return false;
    }

    if (decoded_mat.type() != CV_8UC3 || decoded_mat.channels() != 3) {
        spdlog::warn("ConvertCompressedImage: unsupported image format type={}, channels={}", decoded_mat.type(), decoded_mat.channels());
        return false;
    }

    const int width = decoded_mat.cols;
    const int height = decoded_mat.rows;
    constexpr int kChannels = 3;
    const std::size_t payload_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * kChannels;
    if (payload_size > image_msg.data.size()) {
        spdlog::warn("ConvertCompressedImage: payload overflow, need {}, buffer {}", payload_size, image_msg.data.size());
        return false;
    }

    image_msg.header.timestamp_ns =
        static_cast<uint64_t>(ros_img.header.stamp_sec) * 1000000000ULL + static_cast<uint64_t>(ros_img.header.stamp_nsec);
    image_msg.header.frame_id.fill('\0');
    std::memcpy(image_msg.header.frame_id.data(), ros_img.header.frame_id.data(),
                std::min(ros_img.header.frame_id.size(), image_msg.header.frame_id.size() - 1));
    image_msg.header.encoding.fill('\0');
    constexpr std::string_view kEncoding = "bgr8";
    std::memcpy(image_msg.header.encoding.data(), kEncoding.data(), kEncoding.size());
    image_msg.header.width = static_cast<uint32_t>(width);
    image_msg.header.height = static_cast<uint32_t>(height);
    image_msg.header.step = static_cast<uint32_t>(width * kChannels);
    image_msg.header.payload_size = static_cast<uint32_t>(payload_size);
    image_msg.header.compressed = false;

    std::memcpy(image_msg.data.data(), decoded_mat.data, payload_size);
    return true;
}

/**
 * @brief 解析原始图像并转换为定长图像结构（目前支持 bgr8）
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param image_msg 输出图像
 * @return 成功返回 true
 */
bool ConvertRawImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, ms_slam::slam_common::Image& image_msg)
{
    if (descriptor.kind != MessageKind::Image) {
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
    if (expected_size > ms_slam::slam_common::kImageMaxDataSize) {
        spdlog::warn("ConvertRawImage: payload {} exceeds buffer {}", expected_size, ms_slam::slam_common::kImageMaxDataSize);
        return false;
    }
    if (ros_img.data.size() < expected_size) {
        spdlog::warn("ConvertRawImage: payload too small, have {}, need {}", ros_img.data.size(), expected_size);
        return false;
    }

    image_msg.header.timestamp_ns =
        static_cast<uint64_t>(ros_img.header.stamp_sec) * 1000000000ULL + static_cast<uint64_t>(ros_img.header.stamp_nsec);
    image_msg.header.frame_id.fill('\0');
    std::memcpy(image_msg.header.frame_id.data(), ros_img.header.frame_id.data(),
                std::min(ros_img.header.frame_id.size(), image_msg.header.frame_id.size() - 1));
    image_msg.header.encoding.fill('\0');
    constexpr std::string_view kEncoding = "bgr8";
    std::memcpy(image_msg.header.encoding.data(), kEncoding.data(), kEncoding.size());
    image_msg.header.width = ros_img.width;
    image_msg.header.height = ros_img.height;
    image_msg.header.step = expected_step;
    image_msg.header.payload_size = static_cast<uint32_t>(expected_size);
    image_msg.header.compressed = false;

    std::memcpy(image_msg.data.data(), ros_img.data.data(), expected_size);
    return true;
}

/**
 * @brief 解析 IMU 消息为 LivoxImuData
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param imu_msg 输出 IMU
 * @return 成功返回 true
 */
bool ConvertImu(const mcap::MessageView& view, const MessageDescriptor& descriptor, LivoxImuData& imu_msg)
{
    if (descriptor.kind != MessageKind::Imu) {
        return false;
    }

    ROS1Imu imu;
    if (!imu.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/Imu at topic {}", topic);
        return false;
    }

    imu_msg.timestamp_ns = static_cast<uint64_t>(imu.header.stamp_sec) * 1000000000ULL + imu.header.stamp_nsec;
    imu_msg.index = imu.header.seq;
    imu_msg.angular_velocity = {static_cast<float>(imu.angular_velocity.x), static_cast<float>(imu.angular_velocity.y), static_cast<float>(imu.angular_velocity.z)};
    imu_msg.linear_acceleration = {
        static_cast<float>(imu.linear_acceleration.x),
        static_cast<float>(imu.linear_acceleration.y),
        static_cast<float>(imu.linear_acceleration.z)};
    return true;
}

/**
 * @brief MCAP 播放窗口配置
 */
struct PlaybackOptions {
    bool sync_time{true};          ///< 是否按照时间同步播放
    double playback_rate{1.0};     ///< 播放倍率
    double start_offset_s{0.0};    ///< 起始偏移（秒）
    double duration_s{0.0};        ///< 播放时长（秒，为0表示到文件末尾）
};

/**
 * @brief 从 MCAP 文件读取数据并注入 Odometry
 */
class McapPlaybackRunner {
public:
    /**
     * @brief 构造函数
     * @param odom 里程计实例
     * @param mcap_path MCAP 文件路径
     * @param options 播放配置
     * @param blind_dist 点云盲区距离
     * @param use_image 是否启用图像处理
     * @param exit_flag 全局退出标志
     */
    McapPlaybackRunner(std::shared_ptr<FilterOdometry> odom,
                       std::string mcap_path,
                       PlaybackOptions options,
                       double blind_dist,
                       bool use_image,
                       std::atomic<bool>& exit_flag)
        : odom_(std::move(odom))
        , mcap_path_(std::move(mcap_path))
        , options_(options)
        , blind_dist_(blind_dist)
        , use_image_(use_image)
        , exit_flag_(&exit_flag)
    {
    }

    /**
     * @brief 启动播放线程
     */
    void Start()
    {
        worker_ = std::thread([this]() { Run(); });
    }

    /**
     * @brief 等待线程结束
     */
    void Join()
    {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    /**
     * @brief 线程入口，负责读取 MCAP 并调用 Odometry
     */
    void Run()
    {
        mcap::McapReader reader;
        const auto status = reader.open(mcap_path_);
        if (!status.ok()) {
            spdlog::error("Failed to open MCAP file {}: {}", mcap_path_, status.message);
            exit_flag_->store(true);
            return;
        }

        const auto summary_status = reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
        if (!summary_status.ok()) {
            spdlog::warn("Failed to read MCAP summary: {}", summary_status.message);
        }

        const uint64_t start_offset_ns = options_.start_offset_s > 0.0
                                             ? static_cast<uint64_t>(std::llround(options_.start_offset_s * 1e9))
                                             : 0U;
        const uint64_t duration_ns = options_.duration_s > 0.0 ? static_cast<uint64_t>(std::llround(options_.duration_s * 1e9)) : 0U;
        const double rate = options_.playback_rate > 0.0 ? options_.playback_rate : 1.0;

        std::optional<uint64_t> bag_start_ns;
        std::optional<uint64_t> process_start_ns;
        std::optional<uint64_t> process_end_ns;
        std::optional<std::chrono::steady_clock::time_point> wall_start;

        auto pc_frame = std::make_shared<Mid360Frame>();
        auto image_msg = std::make_shared<ms_slam::slam_common::Image>();
        auto imu_msg = std::make_shared<LivoxImuData>();

        auto messages = reader.readMessages();
        for (auto it = messages.begin(); it != messages.end() && !exit_flag_->load(); ++it) {
            const auto& view = *it;
            const uint64_t timestamp_ns = (view.message.logTime != 0) ? view.message.logTime : view.message.publishTime;

            if (!bag_start_ns.has_value()) {
                bag_start_ns = timestamp_ns;
                process_start_ns = bag_start_ns.value() + start_offset_ns;
                if (duration_ns > 0) {
                    process_end_ns = process_start_ns.value() + duration_ns;
                }
                wall_start = std::chrono::steady_clock::now();
                spdlog::info("MCAP playback started at {:.6f}s", static_cast<double>(timestamp_ns) * 1e-9);
            }

            if (process_start_ns.has_value() && timestamp_ns < process_start_ns.value()) {
                continue;
            }
            if (process_end_ns.has_value() && timestamp_ns > process_end_ns.value()) {
                spdlog::info("Reached configured playback window end, stopping");
                break;
            }

            if (options_.sync_time && bag_start_ns.has_value() && wall_start.has_value()) {
                const double rel_ns = static_cast<double>(timestamp_ns - bag_start_ns.value()) / rate;
                const auto target = wall_start.value() + std::chrono::nanoseconds(static_cast<int64_t>(rel_ns));
                const auto now = std::chrono::steady_clock::now();
                if (target > now) {
                    std::this_thread::sleep_until(target);
                }
            }

            const MessageDescriptor descriptor = DescribeMessage(view);
            switch (descriptor.kind) {
                case MessageKind::PointCloud: {
                    if (!ConvertPointCloud(view, descriptor, *pc_frame)) {
                        spdlog::warn("Skip point cloud due to conversion failure");
                        continue;
                    }
                    auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
                    if (!ConvertMid360Frame(*pc_frame, cloud, blind_dist_)) {
                        spdlog::warn("Failed to convert Mid360Frame to slam_core cloud");
                        continue;
                    }
                    odom_->AddLidarData(cloud);
                    break;
                }
                case MessageKind::CompressedImage: {
                    if (!ConvertCompressedImage(view, descriptor, *image_msg)) {
                        spdlog::warn("Skip compressed image due to conversion failure");
                        continue;
                    }

                    ms_slam::slam_core::Image image;
                    if (!DecodeImageMessage(*image_msg, image)) {
                        spdlog::warn("Failed to decode Image");
                        continue;
                    }
                    if (use_image_) {
                        odom_->AddImageData(image);
                    }
                    break;
                }
                case MessageKind::Image: {
                    if (!ConvertRawImage(view, descriptor, *image_msg)) {
                        spdlog::warn("Skip raw image due to conversion failure");
                        continue;
                    }

                    ms_slam::slam_core::Image image;
                    if (!DecodeImageMessage(*image_msg, image)) {
                        spdlog::warn("Failed to decode Image");
                        continue;
                    }
                    if (use_image_) {
                        odom_->AddImageData(image);
                    }
                    break;
                }
                case MessageKind::Imu: {
                    if (!ConvertImu(view, descriptor, *imu_msg)) {
                        spdlog::warn("Skip IMU due to conversion failure");
                        continue;
                    }
                    IMU imu;
                    if (!ConvertLivoxImuData(*imu_msg, imu)) {
                        spdlog::warn("Failed to convert IMU struct");
                        continue;
                    }
                    odom_->AddIMUData(imu);
                    break;
                }
                case MessageKind::Unsupported:
                default:
                    break;
            }
        }

        reader.close();
        exit_flag_->store(true);
        spdlog::info("MCAP playback finished");
    }

    std::shared_ptr<FilterOdometry> odom_;
    std::string mcap_path_;
    PlaybackOptions options_;
    double blind_dist_{0.5};
    bool use_image_{false};
    std::atomic<bool>* exit_flag_{nullptr};
    std::thread worker_;
};

}  // namespace

/**
 * @brief 程序入口，使用 MCAP 数据驱动 Odometry 并通过 iceoryx2 发布
 * @param argc 参数数量
 * @param argv 参数数组（可选：指定 BagTool 配置路径）
 * @return 进程退出码
 */
int main(int argc, char** argv)
{
    EASY_THREAD_SCOPE("AdapterMcapThread");
    EASY_PROFILER_ENABLE;
    std::signal(SIGINT, HandleSignal);

    LoadConfigFromFile("../config/test.yaml");
    const auto& config_inst = Config::GetInstance();

    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "mcap_sub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("crash_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::from_str(config_inst.common_params.log_level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);

    spdlog::info("Starting SLAM MCAP playback demo");

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("Failed to initialize crash logger!");
        return 1;
    }
    spdlog::info("Crash logger initialized successfully");

    auto node = std::make_shared<IoxNode>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create iceoryx2 node"));
    spdlog::info("iceoryx2 node created for playback pipeline");

    LogConfig();
    const double blind_dist = config_inst.common_params.blind;
    const bool use_img = config_inst.common_params.render_en;

    YAML::Node root = YAML::LoadFile("../../slam_recorder/config/config.yaml");
    auto bag_node = root["BagTool"];
    PlaybackOptions playback_options;
    playback_options.playback_rate = bag_node["playback"]["rate"].as<double>();
    playback_options.start_offset_s = bag_node["time_window"]["start_seconds"].as<double>();
    playback_options.duration_s = bag_node["time_window"]["duration_seconds"].as<double>();
    const std::string input_path = (argc > 1) ? std::string(argv[1]) : bag_node["input"]["path"].as<std::string>();

    auto odom = std::make_shared<FilterOdometry>();

    IoxPublisher<OdomData> odom_pub(node, "/odom");
    IoxPublisher<FrameTransformArray> tf_pub(node, "/tf");
    IoxPublisher<PathData> path_pub(node, "/path");
    IoxPublisher<Mid360Frame> map_cloud_pub(node, "/cloud_registered");
    IoxPublisher<Mid360Frame> local_map_pub(node, "/local_map");

    McapPlaybackRunner playback_runner(odom, input_path, playback_options, blind_dist, use_img, shouldExit);
    playback_runner.Start();

    States lidar_states_buffer;
    std::vector<State> states_buffer;
    OdomData odom_msg{};
    FrameTransformArray tf_msg{};
    PathData path_msg{};
    std::vector<PointCloudType::Ptr> deskewed_clouds;
    PointCloud<PointXYZDescriptor>::Ptr local_map = std::make_shared<PointCloud<PointXYZDescriptor>>();

    while (!shouldExit.load()) {
        EASY_BLOCK("Adapter Publish", profiler::colors::Orange);
        odom->GetLidarState(lidar_states_buffer);

        std::vector<FrameTransformData> transform_data;
        transform_data.reserve(lidar_states_buffer.size() * 2);

        for (const auto& state : lidar_states_buffer) {
            if (!BuildOdomData(state, "odom", "base_link", odom_msg)) {
                spdlog::warn("BuildOdomData failed");
                continue;
            }
            odom_pub.SetBuildCallback([odom_msg](OdomData& payload) { payload = odom_msg; });
            odom_pub.Publish();

            const auto& position = state.p();
            const auto& quat_state = state.quat();

            transform_data.emplace_back(FrameTransformData{
                .timestamp = state.timestamp(),
                .parent_frame = "odom",
                .child_frame = "base_link",
                .translation = position,
                .rotation = quat_state});

            transform_data.emplace_back(FrameTransformData{
                .timestamp = state.timestamp(),
                .parent_frame = "base_link",
                .child_frame = "livox_frame",
                .translation = Eigen::Vector3d(-0.011, -0.02329, 0.04412),
                .rotation = Eigen::Quaterniond::Identity()});

            states_buffer.emplace_back(state);
            if (states_buffer.size() > kMaxPathPoses) {
                states_buffer.erase(states_buffer.begin());
            }
        }

        if (!transform_data.empty()) {
            if (BuildFrameTransformArray(transform_data, tf_msg)) {
                tf_pub.SetBuildCallback([tf_msg](FrameTransformArray& payload) { payload = tf_msg; });
                tf_pub.Publish();
            }
        }

        if (!states_buffer.empty() && states_buffer.size() % 10 == 0 && BuildPathData(states_buffer, "odom", path_msg)) {
            path_pub.SetBuildCallback([path_msg](PathData& payload) { payload = path_msg; });
            path_pub.Publish();
        }

        // 发布 deskew 后地图点云
        odom->GetMapCloud(deskewed_clouds);
        for (const auto& cloud : deskewed_clouds) {
            if (!cloud) {
                continue;
            }
            const double ts_sec = cloud->empty() ? 0.0 : cloud->timestamp(0);
            const uint64_t ts_ns = ts_sec > 0.0 ? static_cast<uint64_t>(std::llround(ts_sec * 1e9)) : 0ULL;
            map_cloud_pub.PublishWithBuilder([&](Mid360Frame& payload) {
                return BuildMid360FrameFromPointCloud(*cloud, ts_ns, payload, "odom");
            });
        }

        // 发布局部地图
        odom->GetLocalMap(local_map);
        if (local_map && local_map->size() > 0) {
            local_map_pub.PublishWithBuilder([&](Mid360Frame& payload) {
                return BuildMid360FrameFromPointCloud(*local_map, 0ULL, payload, "odom");
            });
        }

        EASY_END_BLOCK;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    playback_runner.Join();

    spdlog::info("Exit signal received, cleaning up");

    odom->Stop();

    const auto dumped_blocks = profiler::dumpBlocksToFile(kProfileDumpPath);
    if (dumped_blocks == 0) {
        spdlog::error("Failed to export profiler data: {}", kProfileDumpPath);
    } else {
        spdlog::info("Export profiler data success: {} blocks, path {}", dumped_blocks, kProfileDumpPath);
    }

    return 0;
}
