#include <opencv2/imgcodecs.hpp>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>
#include <slam_common/crash_logger.hpp>
#include <slam_common/callback_dispatcher.hpp>
#include <spdlog/stopwatch.h>
#include <slam_core/odometry.hpp>

#include "slam_adapter/config_loader.hpp"

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
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    LoadConfigFromFile("../config/test.yaml");
    LogConfig();

    auto odom = std::make_unique<Odometry>();

    std::atomic<int> pc_received_count{0};
    auto pc_callback = [&pc_received_count, &odom](const FoxglovePointCloud& pc_wrapper) {
        pc_received_count++;

        // Get native Foxglove PointCloud pointer (zero-copy)
        const foxglove::PointCloud* pc = pc_wrapper.get();

        // Calculate number of points
        size_t num_points = pc->data()->size() / pc->point_stride();

        // Parse point cloud fields
        auto cur_pc = std::make_shared<PointCloud<PointXYZITDescriptor>>();
        cur_pc->reserve(num_points);
        
        const uint32_t x_offset = 0, y_offset = 4, z_offset = 8, intensity_offset = 12, timestamp_offset = 18;

        double timestamp = 0;
        for (std::size_t i = 0; i < num_points; ++i) {
            const auto data = pc->data()->Data() + i * pc->point_stride();
            const float x = *reinterpret_cast<const float*>(data + x_offset);
            const float y = *reinterpret_cast<const float*>(data + y_offset);
            const float z = *reinterpret_cast<const float*>(data + z_offset);
            const float intensity = *reinterpret_cast<const float*>(data + intensity_offset);
            timestamp = *reinterpret_cast<const double*>(data + timestamp_offset);
            cur_pc->push_back(PointXYZIT(x, y, z, intensity, timestamp));
        }

        odom->AddLidarData(cur_pc);
        spdlog::info("✓ Received PointCloud #{}: timestamp={:.3f}, point_stride={}, data_size={}, points={}",
                     pc_received_count.load(),
                     timestamp,
                     pc->point_stride(),
                     pc->data()->size(),
                     cur_pc->size());
    };

    auto pc_subscriber = std::make_shared<FBSSubscriber<FoxglovePointCloud>>(node, "/lidar_points", pc_callback);
    spdlog::info("Starting pointcloud threaded_subscriber...");

    std::atomic<int> received_count{0};
    auto img_callback = [&received_count, &odom](const FoxgloveCompressedImage& img_wrapper) {
        received_count++;

        spdlog::stopwatch ws;
        const foxglove::CompressedImage* img = img_wrapper.get();
        std::vector<uint8_t> buffer(img->data()->begin(), img->data()->end());
        cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
        double timestamp = img->timestamp()->sec() + img->timestamp()->nsec() * 1e-9;

        Image cur_img(mat, timestamp);
        odom->AddImageData(cur_img);

        spdlog::info("✓ Received Image #{}: timestamp={:.3f}, {} bytes, mat size: {}x{}",
                    received_count.load(), timestamp, img->data()->size(), mat.size().width, mat.size().height);

        spdlog::warn("  Decoding time: {} us", std::chrono::duration_cast<std::chrono::microseconds>(ws.elapsed()).count());

        // std::string filename = "received_image.jpg";
        // cv::imwrite(filename, mat);
        // spdlog::info("  Saved to {}", filename);
    };

    auto img_subscriber = std::make_shared<FBSSubscriber<FoxgloveCompressedImage>>(
        node, "/camera/image_raw", img_callback);
    spdlog::info("Starting image threaded_subscriber...");

    std::atomic<int> imu_received_count{0};
    auto imu_callback = [&imu_received_count, &odom](const FoxgloveImu& imu_wrapper) {
        imu_received_count++;

        const foxglove::Imu* imu = imu_wrapper.get();
        double timestamp = imu->timestamp()->sec() + imu->timestamp()->nsec() * 1e-9;

        IMU cur_imu(
            Eigen::Vector3d(imu->angular_velocity()->x(), imu->angular_velocity()->y(), imu->angular_velocity()->z()),
            Eigen::Vector3d(imu->linear_acceleration()->x(), imu->linear_acceleration()->y(), imu->linear_acceleration()->z()),
            timestamp);

        odom->AddIMUData(cur_imu);

        if (imu_received_count.load() % 50 == 0) {
            spdlog::info("✓ Received Imu #{}: timestamp={:.3f}, angular_velocity=({:.3f}, {:.3f}, {:.3f}), linear_acceleration=({:.3f}, {:.3f}, {:.3f})",
                         imu_received_count.load(),
                         cur_imu.timestamp(),
                         cur_imu.angular_velocity().x(), cur_imu.angular_velocity().y(), cur_imu.angular_velocity().z(),
                         cur_imu.linear_acceleration().x(), cur_imu.linear_acceleration().y(), cur_imu.linear_acceleration().z());
        }
    };

    auto imu_subscriber = std::make_shared<FBSSubscriber<FoxgloveImu>>(
        node, "/lidar_imu", imu_callback, PubSubConfig{.subscriber_max_buffer_size = 100});
    spdlog::info("Starting imu threaded_subscriber...");

    CallbackDispatcher dispatcher;
    dispatcher.set_poll_interval(std::chrono::milliseconds(1));
    dispatcher.register_subscriber<FBSSubscriber<FoxglovePointCloud>>(pc_subscriber, "PointCloud_Subscriber", 5);
    dispatcher.register_subscriber<FBSSubscriber<FoxgloveCompressedImage>>(img_subscriber, "Image_Subscriber", 5);
    dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(imu_subscriber, "Imu_Subscriber", 10);
    dispatcher.start();

    std::this_thread::sleep_for(std::chrono::seconds(200));

    dispatcher.stop();
    dispatcher.print_statistics();

    return 0;
}