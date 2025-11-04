#include <atomic>
#include <csignal>

#include <easy/profiler.h>
#include <easy/arbitrary_value.h>
#include <spdlog/stopwatch.h>
#include <slam_common/callback_dispatcher.hpp>
#include <slam_common/crash_logger.hpp>
#include <slam_common/flatbuffers_pub_sub.hpp>
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

    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "sub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("crash_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::from_str(config.log_level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);

    spdlog::info("Starting SLAM test");

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("Failed to initialize crash logger!");
        return 1;
    }
    spdlog::info("Crash logger initialized successfully");

    // Create unique node for subscriber
    auto node =
        std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    spdlog::info("Creating generic publishers and subscribers...");

    LoadConfigFromFile("../config/test.yaml");
    LogConfig();
    const auto& config_inst = Config::GetInstance();

    auto odom = std::make_unique<Odometry>();
    auto processed_image_pub = std::make_shared<FBSPublisher<FoxgloveCompressedImage>>(node, "/camera/image_processed");

    std::atomic<int> pc_received_count{0};
    auto pc_callback = [&pc_received_count, &odom](const FoxglovePointCloud& pc_wrapper) {
        EASY_BLOCK("pc_cb", profiler::colors::Green);
        pc_received_count++;

        // Get native Foxglove PointCloud pointer (zero-copy)
        const foxglove::PointCloud* pc = pc_wrapper.get();

        auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
        if (!ConvertLivoxPointCloudMessage(*pc, cloud)) {
            spdlog::warn("Failed to convert point cloud message");
            return;
        }

        odom->AddLidarData(cloud);
        // spdlog::info(
        //     "✓ Received PointCloud #{}: timestamp={:.3f}, point_stride={}, data_size={}, points={}",
        //     pc_received_count.load(),
        //     cloud->timestamp(0),
        //     pc->point_stride(),
        //     pc->data()->size(),
        //     cloud->size());
    };

    auto pc_subscriber = std::make_shared<FBSSubscriber<FoxglovePointCloud>>(node, config_inst.common_params.lid_topic, pc_callback);
    spdlog::info("Starting pointcloud threaded_subscriber...");

    std::atomic<int> received_count{0};
    auto img_callback = [&received_count, &odom, processed_image_pub](const FoxgloveCompressedImage& img_wrapper) {
        EASY_BLOCK("img_cb", profiler::colors::Coral);
        received_count++;

        spdlog::stopwatch ws;
        const foxglove::CompressedImage* img = img_wrapper.get();
        Image image;
        if (!DecodeCompressedImageMessage(*img, image)) {
            spdlog::warn("Failed to decode compressed image");
            return;
        }

        odom->AddImageData(image);

        if (processed_image_pub) {
            flatbuffers::FlatBufferBuilder image_builder(512 * 1024);
            const std::string frame_id = img->frame_id() ? img->frame_id()->str() : "camera";
            if (BuildFoxgloveCompressedImage(image, frame_id, "jpeg", 90, image_builder)) {
                processed_image_pub->publish_from_builder(image_builder);
            }
        }

        // spdlog::info(
        //     "✓ Received Image #{}: timestamp={:.3f}, {} bytes, mat size: {}x{}",
        //     received_count.load(),
        //     image.timestamp(),
        //     img->data()->size(),
        //     image.data().size().width,
        //     image.data().size().height);

        // spdlog::warn("  Decoding time: {} us", std::chrono::duration_cast<std::chrono::microseconds>(ws.elapsed()).count());

        // std::string filename = "received_image.jpg";
        // cv::imwrite(filename, mat);
        // spdlog::info("  Saved to {}", filename);
    };

    auto img_subscriber = std::make_shared<FBSSubscriber<FoxgloveCompressedImage>>(node, config_inst.common_params.img_topics[0], img_callback);
    spdlog::info("Starting image threaded_subscriber...");

    std::atomic<int> imu_received_count{0};
    auto imu_callback = [&imu_received_count, &odom](const FoxgloveImu& imu_wrapper) {
        EASY_BLOCK("imu_cb", profiler::colors::Brown);
        imu_received_count++;

        const foxglove::Imu* imu = imu_wrapper.get();
        IMU cur_imu;
        if (!ConvertImuMessage(*imu, cur_imu, false)) {
            spdlog::warn("Failed to convert IMU message");
            return;
        }

        odom->AddIMUData(cur_imu);

        // if (imu_received_count.load() % 50 == 0) {
        //     spdlog::info(
        //         "✓ Received Imu #{}: timestamp={:.3f}, angular_velocity=({:.3f}, {:.3f}, {:.3f}), linear_acceleration=({:.3f}, {:.3f}, {:.3f})",
        //         imu_received_count.load(),
        //         cur_imu.timestamp(),
        //         cur_imu.angular_velocity().x(),
        //         cur_imu.angular_velocity().y(),
        //         cur_imu.angular_velocity().z(),
        //         cur_imu.linear_acceleration().x(),
        //         cur_imu.linear_acceleration().y(),
        //         cur_imu.linear_acceleration().z());
        // }
    };

    auto imu_subscriber =
        std::make_shared<FBSSubscriber<FoxgloveImu>>(node, config_inst.common_params.imu_topic, imu_callback, PubSubConfig{.subscriber_max_buffer_size = 100});
    spdlog::info("Starting imu threaded_subscriber...");

    CallbackDispatcher dispatcher;
    dispatcher.set_poll_interval(std::chrono::milliseconds(1));
    dispatcher.register_subscriber<FBSSubscriber<FoxglovePointCloud>>(pc_subscriber, "PointCloud_Subscriber", 5);
    dispatcher.register_subscriber<FBSSubscriber<FoxgloveCompressedImage>>(img_subscriber, "Image_Subscriber", 5);
    dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(imu_subscriber, "Imu_Subscriber", 10);
    dispatcher.start();

    States lidar_states_buffer;
    auto odom_pub = std::make_shared<FBSPublisher<FoxglovePoseInFrame>>(node, "/odom");
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);

    auto tf_pub = std::make_shared<FBSPublisher<FoxgloveFrameTransforms>>(node, "/tf");
    auto scene_pub = std::make_shared<FBSPublisher<FoxgloveSceneUpdate>>(node, "/marker");

    std::vector<PointCloudType::Ptr> deskewed_clouds;
    auto deskewed_cloud_pub = std::make_shared<FBSPublisher<FoxglovePointCloud>>(node, "/deskewed_cloud");

    PointCloud<PointXYZDescriptor>::Ptr local_map;
    local_map = std::make_shared<PointCloud<PointXYZDescriptor>>();
    auto local_map_pub = std::make_shared<FBSPublisher<FoxglovePointCloud>>(node, "/local_map");

    std::vector<State> states_buffer;
    auto path_pub = std::make_shared<FBSPublisher<FoxglovePosesInFrame>>(node, "/path");

    while (!shouldExit.load()) {
        EASY_BLOCK("Adaprer Publish", profiler::colors::Orange);
        // Get latest states from the odometry
        odom->GetLidarState(lidar_states_buffer);

        std::vector<FrameTransformData> transform_data;
        transform_data.reserve(lidar_states_buffer.size() * 2);

        for (const auto& state : lidar_states_buffer) {
            if (!BuildFoxglovePoseInFrame(state, "odom", fbb)) {
                spdlog::warn("BuildFoxglovePoseInFrame failed");
                continue;
            }
            odom_pub->publish_from_builder(fbb);

            if (BuildFoxgloveSceneUpdateFromState(state, "odom", "odom_covariance", fbb)) {
                scene_pub->publish_from_builder(fbb);
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
            if (BuildFoxgloveFrameTransforms(transform_data, fbb)) {
                tf_pub->publish_from_builder(fbb);
            }
        }

        // Deskewed cloud
        odom->GetDeskewedCloud(deskewed_clouds);
        for (const auto& cloud : deskewed_clouds) {
            if (!cloud) {
                continue;
            }
            if (BuildFoxglovePointCloud(*cloud, "livox_frame", fbb)) {
                deskewed_cloud_pub->publish_from_builder(fbb);
            }
        }

        odom->GetLocalMap(local_map);
        if (local_map && BuildFoxglovePointCloud(*local_map, "livox_frame", fbb)) {
            local_map_pub->publish_from_builder(fbb);
        }

        if (states_buffer.size() % 10 == 0 && BuildFoxglovePosesInFrame(states_buffer, "odom", fbb)) {
            path_pub->publish_from_builder(fbb);
        }

        EASY_END_BLOCK;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::info("接收到退出信号，正在收尾");

    dispatcher.stop();
    dispatcher.print_statistics();
    odom->Stop();

    const auto dumped_blocks = profiler::dumpBlocksToFile(kProfileDumpPath);
    if (dumped_blocks == 0) {
        spdlog::error("导出Profiler数据失败: {}", kProfileDumpPath);
    } else {
        spdlog::info("导出Profiler数据成功: {} 块, 路径 {}", dumped_blocks, kProfileDumpPath);
    }

    return 0;
}
