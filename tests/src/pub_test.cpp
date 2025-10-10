#include "slam_common/generic_flatbuffer_pubsub.hpp"
#include "slam_common/foxglove_messages.hpp"

#include <slam_common/slam_crash_logger.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using namespace ms_slam::slam_common;

flatbuffers::DetachedBuffer opencv_mat_to_image(const cv::Mat& mat, uint64_t timestamp, uint32_t seq)
{
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);
    foxglove::Time ts(timestamp / 1000000000, timestamp % 1000000000);

    auto frame_id = fbb.CreateString("camera_frame");

    // Create format
    auto format = fbb.CreateString("jpeg");

    std::vector<uint8_t> image_data;
    bool success = cv::imencode(".jpg", mat, image_data);
    auto data = fbb.CreateVector(image_data);

    // Create CompressedImage
    auto compressed_img = foxglove::CreateCompressedImage(
        fbb,
        &ts,
        frame_id,
        data,
        format
    );

    fbb.Finish(compressed_img);

    return fbb.Release();
}

int main()
{
    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "pub.log";
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

    // Create unique node for publisher
   auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    // Create Image publisher using Handle-based CuImage (now iceoryx2 compatible)
    GenericFlatBufferPublisher<FoxgloveCompressedImage> img_publisher(node, "/test/image");

    // 读取OpenCV图像
    cv::Mat img = cv::imread("/home/ubuntu/data/image.jpg", cv::IMREAD_COLOR);

    auto test_image = opencv_mat_to_image(
        img, std::chrono::steady_clock::now().time_since_epoch().count(), 1);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    img_publisher.publish_raw(test_image.data(), test_image.size());
    spdlog::info("published test image");

    std::this_thread::sleep_for(std::chrono::seconds(5));

    return 0;
}