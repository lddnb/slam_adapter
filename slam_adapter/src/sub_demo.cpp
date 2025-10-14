#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>
#include <slam_common/slam_crash_logger.hpp>
#include <slam_common/callback_dispatcher.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using namespace ms_slam::slam_common;

int main()
{
    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "sub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("slam_crash_logger", dup_filter);

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

    std::atomic<int> pc_received_count{0};
    auto pc_callback = [&pc_received_count](const FoxglovePointCloud& pc_wrapper) {
        pc_received_count++;

        // Get native Foxglove PointCloud pointer (zero-copy)
        const foxglove::PointCloud* pc = pc_wrapper.get();

        spdlog::info("✓ Received PointCloud #{}: frame_id={}, point_stride={}, data_size={}",
                     pc_received_count.load(),
                     pc->frame_id()->c_str(),
                     pc->point_stride(),
                     pc->data()->size());

        // Calculate number of points
        size_t num_points = pc->data()->size() / pc->point_stride();
        spdlog::info("  Number of points: {}", num_points);
    };

    auto pc_subscriber = std::make_shared<FBSSubscriber<FoxglovePointCloud>>(node, "/lidar_points", pc_callback);
    spdlog::info("Starting pointcloud threaded_subscriber...");

    std::atomic<int> received_count{0};
    auto img_callback = [&received_count](const FoxgloveCompressedImage& img_wrapper) {
        received_count++;

        auto start = std::chrono::steady_clock::now();
        const foxglove::CompressedImage* img = img_wrapper.get();
        spdlog::info("✓ Received Image #{}: format={}, {} bytes",
                    received_count.load(), img->format()->c_str(), img->data()->size());
        std::vector<uint8_t> buffer(img->data()->begin(), img->data()->end());
        cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
        spdlog::info(" mat size: {}x{}", mat.size().width, mat.size().height);
        auto duration = std::chrono::steady_clock::now() - start;
        spdlog::warn("  Decoding time: {} us", std::chrono::duration_cast<std::chrono::microseconds>(duration).count());

        // std::string filename = "received_image.jpg";
        // cv::imwrite(filename, mat);
        // spdlog::info("  Saved to {}", filename);
    };

    auto img_subscriber = std::make_shared<FBSSubscriber<FoxgloveCompressedImage>>(
        node, "/camera/image_raw", img_callback);
    spdlog::info("Starting image threaded_subscriber...");

    std::atomic<int> imu_received_count{0};
    auto imu_callback = [&imu_received_count](const FoxgloveImu& imu_wrapper) {
        imu_received_count++;

        const foxglove::Imu* imu = imu_wrapper.get();

        if (imu_received_count.load() % 30 == 0) {
            spdlog::info("✓ Received Imu #{}: frame_id={}, angular_velocity=({:.3f}, {:.3f}, {:.3f}), linear_acceleration=({:.3f}, {:.3f}, {:.3f})",
                         imu_received_count.load(),
                         imu->frame_id()->c_str(),
                         imu->angular_velocity()->x(), imu->angular_velocity()->y(), imu->angular_velocity()->z(),
                         imu->linear_acceleration()->x(), imu->linear_acceleration()->y(), imu->linear_acceleration()->z());
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