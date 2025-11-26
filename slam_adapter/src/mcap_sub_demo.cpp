#include <csignal>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <easy/arbitrary_value.h>
#include <easy/profiler.h>
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <mcap/reader.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <slam_common/crash_logger.hpp>
#include <slam_common/foxglove_messages.hpp>
#include <slam_core/odometry.hpp>

#include "slam_adapter/config_loader.hpp"
#include "slam_adapter/sensor_preprocess.hpp"
#include "slam_adapter/sensor_publish.hpp"
#include "slam_recorder/ros1_msg.hpp"

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
enum class MessageKind { PointCloud, CompressedImage, Imu, Unsupported };

/**
 * @brief 消息特征描述
 */
struct MessageDescriptor {
    MessageKind kind{MessageKind::Unsupported};  ///< 消息类型
    bool is_ros{false};                          ///< 是否为ROS消息
    bool is_protobuf{false};                     ///< 是否为Protobuf消息
    bool is_livox{false};                        ///< 是否为Livox自定义消息
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
    const std::string encoding = view.channel ? view.channel->messageEncoding : "";

    auto mark_ros = [&](MessageKind kind) {
        descriptor.kind = kind;
        descriptor.is_ros = true;
    };

    if (schema_name == "sensor_msgs/PointCloud2") {
        mark_ros(MessageKind::PointCloud);
        return descriptor;
    }
    if (schema_name == "sensor_msgs/CompressedImage") {
        mark_ros(MessageKind::CompressedImage);
        return descriptor;
    }
    if (schema_name == "sensor_msgs/Imu") {
        mark_ros(MessageKind::Imu);
        return descriptor;
    }
    if (schema_name == "livox_ros_driver2/CustomMsg" || schema_name == "livox_ros_driver/CustomMsg") {
        descriptor.kind = MessageKind::PointCloud;
        descriptor.is_ros = true;
        descriptor.is_livox = true;
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
    }

    descriptor.kind = MessageKind::Unsupported;
    return descriptor;
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
 * @brief 解析压缩图像消息
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标压缩图像
 * @return 解析成功返回 true
 */
bool ConvertImage(const mcap::MessageView& view, const MessageDescriptor& descriptor, FoxgloveCompressedImage& message)
{
    if (descriptor.is_protobuf) {
        return ParseProtobufMessage(view, message);
    }

    ROS1CompressedImage ros_img;
    if (!ros_img.Parse(reinterpret_cast<const uint8_t*>(view.message.data), view.message.dataSize)) {
        const std::string topic = view.channel ? view.channel->topic : "<unknown>";
        spdlog::error("Failed to parse sensor_msgs/CompressedImage at topic {}", topic);
        return false;
    }

    message.Clear();
    FillTimestamp(ros_img.header.stamp_sec, ros_img.header.stamp_nsec, *message.mutable_timestamp());
    message.set_frame_id(ros_img.header.frame_id);
    message.set_format(ros_img.format);
    message.set_data(reinterpret_cast<const char*>(ros_img.data.data()), ros_img.data.size());
    return true;
}

/**
 * @brief 解析 IMU 消息
 * @param view MCAP 消息视图
 * @param descriptor 消息描述
 * @param message 目标 IMU
 * @return 解析成功返回 true
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
    message.mutable_angular_velocity()->set_x(imu.angular_velocity.x);
    message.mutable_angular_velocity()->set_y(imu.angular_velocity.y);
    message.mutable_angular_velocity()->set_z(imu.angular_velocity.z);
    message.mutable_linear_acceleration()->set_x(imu.linear_acceleration.x);
    message.mutable_linear_acceleration()->set_y(imu.linear_acceleration.y);
    message.mutable_linear_acceleration()->set_z(imu.linear_acceleration.z);
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
    McapPlaybackRunner(std::shared_ptr<Odometry> odom,
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

        FoxglovePointCloud pc_msg;
        FoxgloveCompressedImage img_msg;
        FoxgloveImu imu_msg;

        auto messages = reader.readMessages();
        for (auto it = messages.begin(); it != messages.end() && !exit_flag_->load(); ++it) {
            const auto& view = *it;
            const uint64_t timestamp_ns = (view.message.logTime != 0) ? view.message.logTime : view.message.publishTime;

            if (!bag_start_ns.has_value()) {
                // 首帧初始化，计算窗口边界与播放起点
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
                // 按录制时间轴和倍率进行节流，保持时序一致
                const double rel_ns = static_cast<double>(timestamp_ns - bag_start_ns.value()) / rate;
                const auto target = wall_start.value() + std::chrono::nanoseconds(static_cast<int64_t>(rel_ns));
                const auto now = std::chrono::steady_clock::now();
                if (target > now) {
                    std::this_thread::sleep_until(target);
                }
            }

            const MessageDescriptor descriptor = DescribeMessage(view);
            if (descriptor.kind == MessageKind::Unsupported) {
                continue;
            }

            switch (descriptor.kind) {
                case MessageKind::PointCloud: {
                    if (!ConvertPointCloud(view, descriptor, pc_msg)) {
                        spdlog::warn("Skip point cloud due to conversion failure");
                        continue;
                    }
#ifdef USE_PCL
                    auto pcl_cloud = std::make_shared<PointCloudT>();
                    if (ConvertLivoxPointCloudMessagePCL(pc_msg, *pcl_cloud)) {
                        odom_->PCLAddLidarData(pcl_cloud);
                    }
#endif
                    auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
                    if (!ConvertLivoxPointCloudMessage(pc_msg, cloud, blind_dist_)) {
                        spdlog::warn("Failed to convert Foxglove point cloud to internal format");
                        continue;
                    }
                    odom_->AddLidarData(cloud);
                    break;
                }
                case MessageKind::CompressedImage: {
                    if (!ConvertImage(view, descriptor, img_msg)) {
                        spdlog::warn("Skip image due to conversion failure");
                        continue;
                    }

                    Image image;
                    if (!DecodeCompressedImageMessage(img_msg, image)) {
                        spdlog::warn("Failed to decode compressed image");
                        continue;
                    }
                    if (use_image_) {
                        odom_->AddImageData(image);
                    }
                    break;
                }
                case MessageKind::Imu: {
                    if (!ConvertImu(view, descriptor, imu_msg)) {
                        spdlog::warn("Skip IMU due to conversion failure");
                        continue;
                    }
                    IMU imu;
                    if (!ConvertImuMessage(imu_msg, imu, false)) {
                        spdlog::warn("Failed to convert IMU message");
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

    std::shared_ptr<Odometry> odom_;
    std::string mcap_path_;
    PlaybackOptions options_;
    double blind_dist_{0.5};
    bool use_image_{false};
    std::atomic<bool>* exit_flag_{nullptr};
    std::thread worker_;
};

}  // namespace

/**
 * @brief 程序入口，使用 MCAP 数据驱动 Odometry 并发布结果
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

    if (eCAL::Initialize("slam_adapter_mcap_demo") != 0) {
        spdlog::warn("eCAL already initialized, continue with existing context");
    }
    spdlog::info("eCAL initialized, creating publishers...");

    LogConfig();
    const double blind_dist = config_inst.common_params.blind;
    const bool use_img = config_inst.common_params.render_en;

    YAML::Node root = YAML::LoadFile("../../slam_recorder/config/config.yaml");
    auto bag_node = root["BagTool"];
    PlaybackOptions playback_options;
    playback_options.playback_rate = bag_node["playback"]["rate"].as<double>();
    playback_options.start_offset_s = bag_node["time_window"]["start_seconds"].as<double>();
    playback_options.duration_s = bag_node["time_window"]["duration_seconds"].as<double>();
    const std::string input_path = bag_node["input"]["path"].as<std::string>();

    auto odom = std::make_shared<Odometry>();

    eCAL::protobuf::CPublisher<FoxgloveCompressedImage> processed_image_pub("/camera/image_processed");
    eCAL::protobuf::CPublisher<FoxglovePoseInFrame> odom_pub("/odom");
    eCAL::protobuf::CPublisher<FoxgloveFrameTransforms> tf_pub("/tf");
    eCAL::protobuf::CPublisher<FoxgloveSceneUpdate> scene_pub("/marker");
    eCAL::protobuf::CPublisher<FoxglovePointCloud> map_cloud_pub("/cloud_registered");
    eCAL::protobuf::CPublisher<FoxglovePointCloud> local_map_pub("/local_map");
    eCAL::protobuf::CPublisher<FoxglovePosesInFrame> path_pub("/path");

    McapPlaybackRunner playback_runner(odom, input_path, playback_options, blind_dist, use_img, shouldExit);
    playback_runner.Start();

    States lidar_states_buffer;
    std::vector<State> states_buffer;
    std::vector<PointCloudType::Ptr> deskewed_clouds;
    PointCloud<PointXYZDescriptor>::Ptr local_map = std::make_shared<PointCloud<PointXYZDescriptor>>();

    FoxglovePoseInFrame odom_msg;
    FoxgloveSceneUpdate scene_msg;
    FoxgloveFrameTransforms tf_msg;
    FoxglovePointCloud cloud_msg;
    FoxglovePosesInFrame path_msg;

    while (!shouldExit.load()) {
        EASY_BLOCK("Adapter Publish", profiler::colors::Orange);
        odom->GetLidarState(lidar_states_buffer);

        std::vector<FrameTransformData> transform_data;
        transform_data.reserve(lidar_states_buffer.size() * 2);

        for (const auto& state : lidar_states_buffer) {
            if (!BuildFoxglovePoseInFrame(state, "odom", odom_msg)) {
                spdlog::warn("BuildFoxglovePoseInFrame failed");
                continue;
            }
            odom_pub.Send(odom_msg);

            if (BuildFoxgloveSceneUpdateFromState(state, "odom", "odom_covariance", scene_msg)) {
                scene_pub.Send(scene_msg);
            }

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
        }

        if (!transform_data.empty()) {
            if (BuildFoxgloveFrameTransforms(transform_data, tf_msg)) {
                tf_pub.Send(tf_msg);
            }
        }

        odom->GetMapCloud(deskewed_clouds);
        for (const auto& cloud : deskewed_clouds) {
            if (!cloud) {
                continue;
            }
            if (BuildFoxglovePointCloud(*cloud, "odom", cloud_msg)) {
                map_cloud_pub.Send(cloud_msg);
            }
        }

        odom->GetLocalMap(local_map);
        if (local_map && BuildFoxglovePointCloud(*local_map, "odom", cloud_msg)) {
            local_map_pub.Send(cloud_msg);
        }

        if (states_buffer.size() % 10 == 0 && BuildFoxglovePosesInFrame(states_buffer, "odom", path_msg)) {
            path_pub.Send(path_msg);
        }

        EASY_END_BLOCK;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    playback_runner.Join();

    spdlog::info("Exit signal received, cleaning up");

    eCAL::Finalize();
    odom->Stop();

    const auto dumped_blocks = profiler::dumpBlocksToFile(kProfileDumpPath);
    if (dumped_blocks == 0) {
        spdlog::error("Failed to export profiler data: {}", kProfileDumpPath);
    } else {
        spdlog::info("Export profiler data success: {} blocks, path {}", dumped_blocks, kProfileDumpPath);
    }

    return 0;
}
