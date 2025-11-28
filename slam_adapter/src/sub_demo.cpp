#include <atomic>
#include <csignal>
#include <chrono>
#include <mutex>

#include <easy/profiler.h>
#include <easy/arbitrary_value.h>
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <slam_common/crash_logger.hpp>
#include <slam_common/foxglove_messages.hpp>
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

    if (eCAL::Initialize("slam_adapter_sub_demo") != 0) {
        spdlog::warn("eCAL already initialized, continue with existing context");
    }
    spdlog::info("eCAL initialized, creating publishers and subscribers...");

    
    LogConfig();
    const double blind_dist = config_inst.common_params.blind;
    const bool use_img = config_inst.common_params.render_en;

    auto odom = std::make_shared<FilterOdometry>();

    eCAL::protobuf::CPublisher<FoxgloveCompressedImage> processed_image_pub("/camera/image_processed");
    eCAL::protobuf::CPublisher<FoxglovePoseInFrame> odom_pub("/odom");
    eCAL::protobuf::CPublisher<FoxgloveFrameTransforms> tf_pub("/tf");
    eCAL::protobuf::CPublisher<FoxgloveSceneUpdate> scene_pub("/marker");
    eCAL::protobuf::CPublisher<FoxglovePointCloud> map_cloud_pub("/cloud_registered");
    eCAL::protobuf::CPublisher<FoxglovePointCloud> local_map_pub("/local_map");
    eCAL::protobuf::CPublisher<FoxglovePosesInFrame> path_pub("/path");

    std::atomic<int> pc_received_count{0};
    auto pc_subscriber = std::make_shared<eCAL::protobuf::CSubscriber<FoxglovePointCloud>>(config_inst.common_params.lid_topic);
    pc_subscriber->SetReceiveCallback([&pc_received_count, &odom, blind_dist](
                                          const eCAL::STopicId&, const FoxglovePointCloud& pc_msg, long long, long long) {
        EASY_BLOCK("pc_cb", profiler::colors::Green);
        pc_received_count++;

#ifdef USE_PCL
        auto pcl_cloud = std::make_shared<PointCloudT>();
        if (!ConvertLivoxPointCloudMessagePCL(pc_msg, *pcl_cloud)) {
            spdlog::warn("Failed to convert point cloud message to PCL cloud");
            return;
        }
        odom->PCLAddLidarData(pcl_cloud);
#endif

        auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
        if (!ConvertLivoxPointCloudMessage(pc_msg, cloud, blind_dist)) {
            spdlog::warn("Failed to convert point cloud message");
            return;
        }

        odom->AddLidarData(cloud);
    });
    spdlog::info("PointCloud subscriber started on topic {}", config_inst.common_params.lid_topic);

    std::atomic<int> received_count{0};
    std::shared_ptr<eCAL::protobuf::CSubscriber<FoxgloveCompressedImage>> img_subscriber;
    if (!config_inst.common_params.img_topics.empty()) {
        img_subscriber = std::make_shared<eCAL::protobuf::CSubscriber<FoxgloveCompressedImage>>(config_inst.common_params.img_topics[0]);
        img_subscriber->SetReceiveCallback([&received_count, &odom, use_img, &processed_image_pub](
                                               const eCAL::STopicId&, const FoxgloveCompressedImage& img_msg, long long, long long) {
            EASY_BLOCK("img_cb", profiler::colors::Coral);
            received_count++;

            spdlog::stopwatch ws;
            Image image;
            if (!DecodeCompressedImageMessage(img_msg, image)) {
                spdlog::warn("Failed to decode compressed image");
                return;
            }

            if (use_img) {
                odom->AddImageData(image);
            }

            FoxgloveCompressedImage processed_msg;
            const std::string frame_id = img_msg.frame_id().empty() ? "camera" : img_msg.frame_id();
            if (BuildFoxgloveCompressedImage(image, frame_id, "jpeg", 90, processed_msg)) {
                processed_image_pub.Send(processed_msg);
            }

            spdlog::debug("Image cb elapsed {} us", std::chrono::duration_cast<std::chrono::microseconds>(ws.elapsed()).count());
        });
        spdlog::info("Image subscriber started on topic {}", config_inst.common_params.img_topics[0]);
    } else {
        spdlog::warn("No image topic configured, skip image subscriber");
    }

    std::atomic<int> imu_received_count{0};
    auto imu_subscriber = std::make_shared<eCAL::protobuf::CSubscriber<FoxgloveImu>>(config_inst.common_params.imu_topic);
    imu_subscriber->SetReceiveCallback([&imu_received_count, &odom](const eCAL::STopicId&, const FoxgloveImu& imu_msg, long long, long long) {
        imu_received_count++;

        IMU cur_imu;
        if (!ConvertImuMessage(imu_msg, cur_imu, false)) {
            spdlog::warn("Failed to convert IMU message");
            return;
        }

        odom->AddIMUData(cur_imu);
    });
    spdlog::info("IMU subscriber started on topic {}", config_inst.common_params.imu_topic);

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

        // Deskewed cloud
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

    spdlog::info("接收到退出信号，正在收尾");

    eCAL::Finalize();
    odom->Stop();

    const auto dumped_blocks = profiler::dumpBlocksToFile(kProfileDumpPath);
    if (dumped_blocks == 0) {
        spdlog::error("导出Profiler数据失败: {}", kProfileDumpPath);
    } else {
        spdlog::info("导出Profiler数据成功: {} 块, 路径 {}", dumped_blocks, kProfileDumpPath);
    }

    return 0;
}
