#include "slam_common/flatbuffers_pub_sub.hpp"
#include "slam_common/foxglove_messages.hpp"
#include <slam_common/slam_crash_logger.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using namespace ms_slam::slam_common;

cv::Mat image_to_opencv_mat(const foxglove::CompressedImage& img)
{
    cv::Mat mat(1837, 1377, CV_8UC3);
    std::memcpy(mat.data, img.data(), img.data()->size());

    return mat;
}

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

    std::atomic<int> received_count{0};
    auto callback = [&received_count](const FoxgloveCompressedImage& img_wrapper) {
        received_count++;

        const foxglove::CompressedImage* img = img_wrapper.get();
        spdlog::info("✓ [Non-threaded] Received Image #{}: format={}, {} bytes",
                    received_count.load(), img->format()->c_str(), img->data()->size());
        // cv::Mat mat = image_to_opencv_mat(*img);
        std::vector<uint8_t> buffer(img->data()->begin(), img->data()->end());
        cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
        spdlog::info(" mat size: {}x{}", mat.size().width, mat.size().height);

        std::string filename = "received_image.jpg";
        cv::imwrite(filename, mat);
        spdlog::info("  Saved to {}", filename);
    };

    // FBSSubscriber<Image> img_subscriber(node, "/test/image", callback);

    // std::this_thread::sleep_for(std::chrono::seconds(3));
    // img_subscriber.receive_once();

    // std::this_thread::sleep_for(std::chrono::seconds(5));

    // ThreadedSubscriber
    ThreadedFBSSubscriber<FoxgloveCompressedImage> threaded_subscriber(
        node, "/test/image", callback, std::chrono::milliseconds(10));

    threaded_subscriber.start();
    spdlog::info("Starting threaded_subscriber...");

    std::this_thread::sleep_for(std::chrono::seconds(10));

    threaded_subscriber.stop();

    return 0;
}