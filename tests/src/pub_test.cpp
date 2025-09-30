#include "slam_common/generic_flatbuffer_pubsub.hpp"
#include "slam_common/flatbuffer_serializer.hpp"
#include <slam_common/slam_crash_logger.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using namespace ms_slam::slam_common;

Image opencv_mat_to_image(const cv::Mat& mat, uint64_t timestamp, uint32_t seq)
{
    Image img;
    img.timestamp = timestamp;
    img.seq = seq;
    img.format.width = mat.cols;
    img.format.height = mat.rows;
    img.format.stride = mat.step[0];

    // 判断像素格式
    if (mat.channels() == 1) {
        img.format.pixel_format = PIXEL_FORMAT_GRAY8;
    } else if (mat.channels() == 3) {
        img.format.pixel_format = PIXEL_FORMAT_BGR8;
    } else if (mat.channels() == 4) {
        img.format.pixel_format = PIXEL_FORMAT_BGRA8;
    } else {
        img.format.pixel_format = PIXEL_FORMAT_UNKNOWN;
    }

    // 复制数据
    size_t total_bytes = mat.total() * mat.elemSize();
    img.data.resize(total_bytes);
    std::memcpy(img.data.data(), mat.data, total_bytes);

    spdlog::info("opencv_mat_to_image called with mat size: {}x{}, channels: {}",
                 mat.cols, mat.rows, mat.channels());
    spdlog::info("Created Image: {}x{}, {} bytes",
                 img.format.width, img.format.height, img.data.size());

    return img;
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
    GenericFlatBufferPublisher<Image> img_publisher(node, "/test/image");

    // 读取OpenCV图像
    cv::Mat img = cv::imread("/home/ubuntu/data/image.png", cv::IMREAD_COLOR);

    auto test_image = opencv_mat_to_image(
        img, std::chrono::steady_clock::now().time_since_epoch().count(), 1);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    img_publisher.publish(test_image);
    spdlog::info("published test image");

    std::this_thread::sleep_for(std::chrono::seconds(5));

    return 0;
}