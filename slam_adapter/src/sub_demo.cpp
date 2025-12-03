#include <atomic>
#include <csignal>
#include <chrono>
#include <mutex>
#include <thread>

#include <easy/profiler.h>
#include <easy/arbitrary_value.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <slam_common/crash_logger.hpp>
#include <slam_common/iceoryx_pub_sub.hpp>
#include <slam_common/sensor_struct.hpp>
#include <slam_core/odometry.hpp>

#include "slam_adapter/config_loader.hpp"
#include "slam_adapter/sensor_publish.hpp"
#include "slam_adapter/sensor_preprocess.hpp"

using namespace ms_slam::slam_common;
using namespace ms_slam::slam_core;
using namespace ms_slam::slam_adapter;

namespace
{

std::atomic<bool> shouldExit{false};
constexpr char kProfileDumpPath[] = "/home/ubuntu/data/test_profile.prof";

/**
 * @brief 处理SIGINT信号并设置退出标记
 * @param signal 捕获到的信号编号
 */
void HandleSignal(int signal)
{
    (void)signal;
    shouldExit.store(true);
}

}  // namespace

int main()
{
    EASY_THREAD_SCOPE("AdapterThread");
    EASY_PROFILER_ENABLE;
    std::signal(SIGINT, HandleSignal);

    LoadConfigFromFile("../config/test.yaml");
    const auto& config_inst = Config::GetInstance();

    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "sub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("crash_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::from_str(config_inst.common_params.log_level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);

    spdlog::info("Starting SLAM test");

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("Failed to initialize crash logger!");
        return 1;
    }
    spdlog::info("Crash logger initialized successfully");

    auto node = std::make_shared<ms_slam::slam_common::IoxNode>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create iceoryx2 node"));
    spdlog::info("iceoryx2 node created, setting up publishers/subscribers...");

    LogConfig();
    const double blind_dist = config_inst.common_params.blind;
    const bool use_img = config_inst.common_params.render_en;

    auto odom = std::make_shared<FilterOdometry>();

    ms_slam::slam_common::IoxPublisher<ms_slam::slam_common::OdomData> odom_pub(node, "/odom");
    ms_slam::slam_common::IoxPublisher<ms_slam::slam_common::FrameTransformArray> tf_pub(node, "/tf");
    ms_slam::slam_common::IoxPublisher<ms_slam::slam_common::PathData> path_pub(node, "/path");
    ms_slam::slam_common::IoxPublisher<ms_slam::slam_common::Mid360Frame> map_cloud_pub(node, "/cloud_registered");
    ms_slam::slam_common::IoxPublisher<ms_slam::slam_common::Mid360Frame> local_map_pub(node, "/local_map");

    auto pc_subscriber = std::make_shared<ms_slam::slam_common::IoxSubscriber<ms_slam::slam_common::Mid360Frame>>(
        node,
        config_inst.common_params.lid_topic,
        [&odom, blind_dist](const ms_slam::slam_common::Mid360Frame& frame) {
            EASY_BLOCK("pc_cb", profiler::colors::Green);
            auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
            if (!ConvertMid360Frame(frame, cloud, blind_dist)) {
                spdlog::warn("Failed to convert Mid360 frame to point cloud");
                return;
            }
            odom->AddLidarData(cloud);
        });
    spdlog::info("Mid360 subscriber started on service {}", config_inst.common_params.lid_topic);

    std::shared_ptr<ms_slam::slam_common::IoxSubscriber<ms_slam::slam_common::Image>> img_subscriber;
    if (!config_inst.common_params.img_topics.empty()) {
        img_subscriber = std::make_shared<ms_slam::slam_common::IoxSubscriber<ms_slam::slam_common::Image>>(
            node,
            config_inst.common_params.img_topics[0],
            [&odom, use_img](const ms_slam::slam_common::Image& img_msg) {
                EASY_BLOCK("img_cb", profiler::colors::Coral);
                ms_slam::slam_core::Image image;
                if (!DecodeImageMessage(img_msg, image)) {
                    spdlog::warn("Failed to decode Image message");
                    return;
                }
                if (use_img) {
                    odom->AddImageData(image);
                }
            });
        spdlog::info("Image subscriber started on service {}", config_inst.common_params.img_topics[0]);
    } else {
        spdlog::warn("No image topic configured, skip image subscriber");
    }

    auto imu_subscriber = std::make_shared<ms_slam::slam_common::IoxSubscriber<ms_slam::slam_common::LivoxImuData>>(
        node,
        config_inst.common_params.imu_topic,
        [&odom](const ms_slam::slam_common::LivoxImuData& imu_msg) {
            IMU cur_imu;
            if (!ConvertLivoxImuData(imu_msg, cur_imu)) {
                spdlog::warn("Failed to convert IMU message");
                return;
            }
            odom->AddIMUData(cur_imu);
        }, IoxPubSubConfig{.subscriber_max_buffer_size = 500});
    spdlog::info("IMU subscriber started on service {}", config_inst.common_params.imu_topic);

    States lidar_states_buffer;
    std::vector<State> states_buffer;
    ms_slam::slam_common::OdomData odom_msg{};
    ms_slam::slam_common::FrameTransformArray tf_msg{};
    ms_slam::slam_common::PathData path_msg{};
    std::vector<PointCloudType::Ptr> deskewed_clouds;
    PointCloud<PointXYZDescriptor>::Ptr local_map = std::make_shared<PointCloud<PointXYZDescriptor>>();

    while (!shouldExit.load()) {
        EASY_BLOCK("Adapter Publish", profiler::colors::Orange);
        pc_subscriber->ReceiveAll();
        imu_subscriber->ReceiveAll();
        if (img_subscriber) {
            img_subscriber->ReceiveAll();
        }

        odom->GetLidarState(lidar_states_buffer);

        std::vector<FrameTransformData> transform_data;
        transform_data.reserve(lidar_states_buffer.size() * 2);

        for (const auto& state : lidar_states_buffer) {
            if (!BuildOdomData(state, "odom", "base_link", odom_msg)) {
                spdlog::warn("BuildOdomData failed");
                continue;
            }
            odom_pub.SetBuildCallback([odom_msg](ms_slam::slam_common::OdomData& payload) { payload = odom_msg; });
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
            if (states_buffer.size() > ms_slam::slam_common::kMaxPathPoses) {
                states_buffer.erase(states_buffer.begin());
            }
        }

        if (!transform_data.empty()) {
            if (BuildFrameTransformArray(transform_data, tf_msg)) {
                tf_pub.SetBuildCallback([tf_msg](ms_slam::slam_common::FrameTransformArray& payload) { payload = tf_msg; });
                tf_pub.Publish();
            }
        }

        if (!states_buffer.empty() && states_buffer.size() % 10 == 0 && BuildPathData(states_buffer, "odom", path_msg)) {
            path_pub.SetBuildCallback([path_msg](ms_slam::slam_common::PathData& payload) { payload = path_msg; });
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
            map_cloud_pub.PublishWithBuilder([&](ms_slam::slam_common::Mid360Frame& payload) {
                return BuildMid360FrameFromPointCloud(*cloud, ts_ns, payload, "odom");
            });
        }

        // 发布局部地图
        odom->GetLocalMap(local_map);
        if (local_map && local_map->size() > 0) {
            local_map_pub.PublishWithBuilder([&](ms_slam::slam_common::Mid360Frame& payload) {
                return BuildMid360FrameFromPointCloud(*local_map, 0ULL, payload, "odom");
            });
        }

        EASY_END_BLOCK;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::info("接收到退出信号，正在收尾");

    odom->Stop();

    const auto dumped_blocks = profiler::dumpBlocksToFile(kProfileDumpPath);
    if (dumped_blocks == 0) {
        spdlog::error("导出Profiler数据失败: {}", kProfileDumpPath);
    } else {
        spdlog::info("导出Profiler数据成功: {} 块, 路径 {}", dumped_blocks, kProfileDumpPath);
    }

    return 0;
}
