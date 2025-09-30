#include "slam_common/generic_flatbuffer_pubsub.hpp"
#include "slam_common/flatbuffer_serializer.hpp"
#include <slam_common/slam_crash_logger.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using namespace ms_slam::slam_common;

cv::Mat image_to_opencv_mat(const Image& img)
{
    int cv_type;
    switch (img.format.pixel_format) {
        case PIXEL_FORMAT_GRAY8:
            cv_type = CV_8UC1;
            break;
        case PIXEL_FORMAT_RGB8:
        case PIXEL_FORMAT_BGR8:
            cv_type = CV_8UC3;
            break;
        case PIXEL_FORMAT_BGRA8:
            cv_type = CV_8UC4;
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }

    cv::Mat mat(img.format.height, img.format.width, cv_type);
    std::memcpy(mat.data, img.data.data(), img.data.size());

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
    auto callback = [&received_count](const Image& image) {
        received_count++;
        spdlog::info("✓ [Non-threaded] Received Image #{}: seq={}, size={}x{}, {} bytes",
                    received_count.load(), image.seq,
                    image.format.width, image.format.height, image.data.size());
        cv::Mat mat = image_to_opencv_mat(image);
        std::string filename = "received_image.png";
        cv::imwrite(filename, mat);
        spdlog::info("  Saved to {}", filename);
    };

    // GenericFlatBufferSubscriber<Image> img_subscriber(node, "/test/image", callback);

    // std::this_thread::sleep_for(std::chrono::seconds(3));
    // img_subscriber.receive_once();

    // std::this_thread::sleep_for(std::chrono::seconds(5));

    // ThreadedSubscriber
    ThreadedFlatBufferSubscriber<Image> threaded_subscriber(
        node, "/test/image", callback, std::chrono::milliseconds(10));

    threaded_subscriber.start();
    spdlog::info("Starting threaded_subscriber...");

    std::this_thread::sleep_for(std::chrono::seconds(10));

    threaded_subscriber.stop();

    return 0;
}