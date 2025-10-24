#include <opencv2/imgcodecs.hpp>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>
#include <slam_common/crash_logger.hpp>
#include <slam_common/callback_dispatcher.hpp>
#include <spdlog/stopwatch.h>
#include <slam_core/odometry.hpp>

#include "slam_adapter/config_loader.hpp"
#include "slam_adapter/sensor_preprocess.hpp"

using namespace ms_slam::slam_common;
using namespace ms_slam::slam_core;
using namespace ms_slam::slam_adapter;

int main()
{
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
        spdlog::error("❌ Failed to initialize crash logger!");
        return 1;
    }
    spdlog::info("Crash logger initialized successfully");

    // Create unique node for subscriber
    auto node =
        std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    LoadConfigFromFile("../config/test.yaml");
    LogConfig();

    auto odom = std::make_unique<Odometry>();

    std::atomic<int> pc_received_count{0};
    auto pc_callback = [&pc_received_count, &odom](const FoxglovePointCloud& pc_wrapper) {
        pc_received_count++;

        // Get native Foxglove PointCloud pointer (zero-copy)
        const foxglove::PointCloud* pc = pc_wrapper.get();

        auto cloud = std::make_shared<PointCloud<PointXYZITDescriptor>>();
        if (!ConvertLivoxPointCloudMessage(*pc, cloud)) {
            spdlog::warn("⚠️ Failed to convert point cloud message");
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

    auto pc_subscriber = std::make_shared<FBSSubscriber<FoxglovePointCloud>>(node, "/lidar_points", pc_callback);
    spdlog::info("Starting pointcloud threaded_subscriber...");

    std::atomic<int> received_count{0};
    auto img_callback = [&received_count, &odom](const FoxgloveCompressedImage& img_wrapper) {
        received_count++;

        spdlog::stopwatch ws;
        const foxglove::CompressedImage* img = img_wrapper.get();
        Image image;
        if (!DecodeCompressedImageMessage(*img, image)) {
            spdlog::warn("⚠️ Failed to decode compressed image");
            return;
        }

        odom->AddImageData(image);

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

    auto img_subscriber = std::make_shared<FBSSubscriber<FoxgloveCompressedImage>>(node, "/camera/image_raw", img_callback);
    spdlog::info("Starting image threaded_subscriber...");

    std::atomic<int> imu_received_count{0};
    auto imu_callback = [&imu_received_count, &odom](const FoxgloveImu& imu_wrapper) {
        imu_received_count++;

        const foxglove::Imu* imu = imu_wrapper.get();
        IMU cur_imu;
        if (!ConvertImuMessage(*imu, cur_imu, false)) {
            spdlog::warn("⚠️ Failed to convert IMU message");
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
        std::make_shared<FBSSubscriber<FoxgloveImu>>(node, "/lidar_imu", imu_callback, PubSubConfig{.subscriber_max_buffer_size = 100});
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
    std::vector<flatbuffers::Offset<foxglove::FrameTransform>> transform_offsets;
    flatbuffers::FlatBufferBuilder tf_fbb(1024 * 1024);

    while (true) {
        // Get latest states from the odometry
        odom->GetLidarState(lidar_states_buffer);

        transform_offsets.reserve(lidar_states_buffer.size());
        tf_fbb.Clear();
        for (const auto& state : lidar_states_buffer) {
            fbb.Clear();

            const double timestamp_sec = state.timestamp();
            const std::uint32_t sec = static_cast<std::uint32_t>(timestamp_sec);
            const std::uint32_t nsec = static_cast<std::uint32_t>(std::round((timestamp_sec - sec) * 1e9));
            foxglove::Time timestamp(sec, nsec);

            const std::string frame_source = "odom";
            auto frame_id = fbb.CreateString(frame_source);

            const auto& position = state.p();
            const auto& quat_state = state.quat();

            auto pos_vec = foxglove::CreateVector3(fbb, position.x(), position.y(), position.z());
            auto quat = foxglove::CreateQuaternion(fbb, quat_state.x(), quat_state.y(), quat_state.z(), quat_state.w());
            auto pose = foxglove::CreatePose(fbb, pos_vec, quat);
            auto pose_in_frame = foxglove::CreatePoseInFrame(fbb, &timestamp, frame_id, pose);
            fbb.Finish(pose_in_frame);
            odom_pub->publish_from_builder(fbb);

            auto parent = tf_fbb.CreateString("odom");
            auto child = tf_fbb.CreateString("base_link");
            auto translation = foxglove::CreateVector3(tf_fbb, position.x(), position.y(), position.z());
            auto rotation = foxglove::CreateQuaternion(tf_fbb, quat_state.x(), quat_state.y(), quat_state.z(), quat_state.w());

            transform_offsets.emplace_back(foxglove::CreateFrameTransform(tf_fbb, &timestamp, parent, child, translation, rotation));

            // static tf
            auto T_imu_lidar_parent = tf_fbb.CreateString("base_link");
            auto T_imu_lidar_child = tf_fbb.CreateString("livox_frame");
            auto T_imu_lidar_translation = foxglove::CreateVector3(tf_fbb, -0.011, -0.02329, 0.04412);
            auto T_imu_lidar_rotation = foxglove::CreateQuaternion(tf_fbb, 0, 0, 0, 1);
            transform_offsets.emplace_back(foxglove::CreateFrameTransform(tf_fbb, &timestamp, T_imu_lidar_parent, T_imu_lidar_child, T_imu_lidar_translation, T_imu_lidar_rotation));
        }
        auto transforms_vector = tf_fbb.CreateVector(transform_offsets);
        auto frame_transforms = foxglove::CreateFrameTransforms(tf_fbb, transforms_vector);
        tf_fbb.Finish(frame_transforms);
        tf_pub->publish_from_builder(tf_fbb);
        transform_offsets.clear();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    dispatcher.stop();
    dispatcher.print_statistics();

    return 0;
}
