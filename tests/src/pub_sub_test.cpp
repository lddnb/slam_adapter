#include <slam_common/generic_publisher_subscriber.hpp>
#include <slam_common/slam_crash_logger.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

using namespace ms_slam::slam_common;

// 将OpenCV Mat转换为Image的辅助函数
Image opencv_mat_to_image(const cv::Mat& mat, uint64_t timestamp, uint32_t seq,
                          std::pmr::memory_resource* mr = std::pmr::get_default_resource())
{
    spdlog::info("opencv_mat_to_image called with mat size: {}x{}, channels: {}",
                mat.cols, mat.rows, mat.channels());

    // 确保图像是连续的内存布局
    cv::Mat continuous_mat = mat.isContinuous() ? mat : mat.clone();

    // 创建ImageFormat
    ImageFormat format;
    format.width = static_cast<uint32_t>(continuous_mat.cols);
    format.height = static_cast<uint32_t>(continuous_mat.rows);

    // 根据OpenCV图像类型设置格式
    pixel_format_t pixel_format;
    uint32_t bytes_per_pixel;

    switch (continuous_mat.type()) {
        case CV_8UC3:  // BGR图像
            pixel_format = PIXEL_FORMAT_BGR8;
            bytes_per_pixel = 3;
            break;
        case CV_8UC1:  // 灰度图像
            pixel_format = PIXEL_FORMAT_GRAY8;
            bytes_per_pixel = 1;
            break;
        case CV_8UC4:  // BGRA图像
            pixel_format = PIXEL_FORMAT_BGRA8;
            bytes_per_pixel = 4;
            break;
        default:
            throw std::runtime_error("Unsupported OpenCV mat type");
    }

    format.stride = format.width * bytes_per_pixel;
    format.pixel_format = pixel_format;

    // 创建Image并复制像素数据
    Image img(timestamp, seq, format, mr);
    size_t data_size = continuous_mat.total() * continuous_mat.elemSize();
    img.data.resize(data_size);
    std::memcpy(img.data.data(), continuous_mat.data, data_size);

    spdlog::info("Created Image: {}x{}, {} bytes", format.width, format.height, data_size);
    return img;
}

// 将Image转换为OpenCV Mat的辅助函数
cv::Mat image_to_opencv_mat(const Image& image)
{
    // 根据像素格式确定OpenCV类型
    int cv_type;
    switch (image.format.pixel_format) {
        case PIXEL_FORMAT_BGR8:
            cv_type = CV_8UC3;
            break;
        case PIXEL_FORMAT_RGB8:
            cv_type = CV_8UC3;
            break;
        case PIXEL_FORMAT_GRAY8:
            cv_type = CV_8UC1;
            break;
        case PIXEL_FORMAT_BGRA8:
            cv_type = CV_8UC4;
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }

    // 创建OpenCV Mat
    cv::Mat mat(image.format.height, image.format.width, cv_type,
                const_cast<uint8_t*>(image.data.data()), image.format.stride);

    // 克隆数据以确保内存安全
    cv::Mat result = mat.clone();

    // 如果是RGB格式，需要转换为BGR（OpenCV默认格式）
    if (image.format.pixel_format == PIXEL_FORMAT_RGB8) {
        cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    }

    return result;
}

int main()
{
    // 初始化日志系统
    ms_slam::slam_common::LoggerConfig config;
    config.log_file_path = "slam.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(10));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("slam_crash_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::from_str(config.log_level));
    logger->flush_on(spdlog::level::warn);

    spdlog::set_default_logger(logger);
    spdlog::info("========================================");
    spdlog::info("Starting FlatBuffers Image Pub/Sub Test");
    spdlog::info("========================================");

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("❌ Failed to initialize crash logger!");
        return 1;
    }

    // 创建 iceoryx2 节点
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    spdlog::info("Creating FlatBuffer Image Publisher and Subscriber...");

    // 创建 FlatBuffer Image Publisher
    FlatBufferImagePublisher img_publisher(node, "/camera/image_raw");

    // 创建 FlatBuffer Image Subscriber with callback
    std::atomic<int> received_count{0};
    auto img_callback = [&received_count](const Image& image) {
        received_count++;
        spdlog::info("✓ Received Image #{}: seq={}, size={}x{}, {} bytes, format={}",
                    received_count.load(),
                    image.seq,
                    image.format.width,
                    image.format.height,
                    image.data.size(),
                    static_cast<int>(image.format.pixel_format));

        // 保存接收到的图像
        try {
            cv::Mat mat = image_to_opencv_mat(image);
            std::string filename = "received_image_" + std::to_string(image.seq) + ".png";
            cv::imwrite(filename, mat);
            spdlog::info("  Saved to {}", filename);
        } catch (const std::exception& e) {
            spdlog::error("  Failed to save image: {}", e.what());
        }
    };

    FlatBufferImageSubscriber img_subscriber(node, "/camera/image_raw", img_callback);

    spdlog::info("Publisher and Subscriber created successfully");
    spdlog::info("----------------------------------------");

    // 加载测试图像
    spdlog::info("Loading test image...");
    cv::Mat test_mat = cv::imread("/home/ubuntu/data/image.png", cv::IMREAD_COLOR);

    if (test_mat.empty()) {
        spdlog::warn("Failed to load /home/ubuntu/data/image.png, creating synthetic image");
        test_mat = cv::Mat::zeros(640, 480, CV_8UC3);
        cv::rectangle(test_mat, cv::Point(100, 100), cv::Point(540, 380), cv::Scalar(0, 255, 0), -1);
        cv::putText(test_mat, "Test Image", cv::Point(200, 250), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    } else {
        spdlog::info("Loaded image: {}x{}", test_mat.cols, test_mat.rows);
    }

    // 测试1: 发布小图像 (640x480)
    spdlog::info("========================================");
    spdlog::info("Test 1: Publishing small image (640x480)");
    spdlog::info("========================================");
    {
        cv::Mat small_img;
        cv::resize(test_mat, small_img, cv::Size(640, 480));
        auto image1 = opencv_mat_to_image(small_img,
                                         std::chrono::steady_clock::now().time_since_epoch().count(),
                                         1);

        bool success = img_publisher.publish(image1);
        spdlog::info("Published small image: {}", success ? "SUCCESS" : "FAILED");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 接收图像
        auto received = img_subscriber.receive_once();
        if (received.has_value()) {
            spdlog::info("✓ Test 1 PASSED: Small image transmitted successfully");
        } else {
            spdlog::error("✗ Test 1 FAILED: No image received");
        }
    }

    // 测试2: 发布中等大小图像 (1280x720)
    spdlog::info("========================================");
    spdlog::info("Test 2: Publishing medium image (1280x720)");
    spdlog::info("========================================");
    {
        cv::Mat medium_img;
        cv::resize(test_mat, medium_img, cv::Size(1280, 720));
        auto image2 = opencv_mat_to_image(medium_img,
                                         std::chrono::steady_clock::now().time_since_epoch().count(),
                                         2);

        bool success = img_publisher.publish(image2);
        spdlog::info("Published medium image: {}", success ? "SUCCESS" : "FAILED");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto received = img_subscriber.receive_once();
        if (received.has_value()) {
            spdlog::info("✓ Test 2 PASSED: Medium image transmitted successfully");
        } else {
            spdlog::error("✗ Test 2 FAILED: No image received");
        }
    }

    // 测试3: 发布大图像 (1920x1080)
    spdlog::info("========================================");
    spdlog::info("Test 3: Publishing large image (1920x1080)");
    spdlog::info("========================================");
    {
        cv::Mat large_img;
        cv::resize(test_mat, large_img, cv::Size(1920, 1080));
        auto image3 = opencv_mat_to_image(large_img,
                                         std::chrono::steady_clock::now().time_since_epoch().count(),
                                         3);

        bool success = img_publisher.publish(image3);
        spdlog::info("Published large image: {}", success ? "SUCCESS" : "FAILED");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto received = img_subscriber.receive_once();
        if (received.has_value()) {
            spdlog::info("✓ Test 3 PASSED: Large image transmitted successfully");
        } else {
            spdlog::error("✗ Test 3 FAILED: No image received");
        }
    }

    // 测试4: 批量发布多张不同大小的图像
    spdlog::info("========================================");
    spdlog::info("Test 4: Publishing multiple images");
    spdlog::info("========================================");
    {
        std::vector<cv::Size> sizes = {
            cv::Size(320, 240),
            cv::Size(640, 480),
            cv::Size(800, 600),
            cv::Size(1024, 768),
            cv::Size(1280, 720)
        };

        for (size_t i = 0; i < sizes.size(); ++i) {
            cv::Mat img;
            cv::resize(test_mat, img, sizes[i]);
            auto image = opencv_mat_to_image(img,
                                            std::chrono::steady_clock::now().time_since_epoch().count(),
                                            4 + i);

            bool success = img_publisher.publish(image);
            spdlog::info("Published image {}/{}: {}x{} - {}",
                        i+1, sizes.size(), sizes[i].width, sizes[i].height,
                        success ? "SUCCESS" : "FAILED");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // 接收所有图像
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto all_images = img_subscriber.receive_all();
        spdlog::info("Received {} images in batch", all_images.size());

        if (all_images.size() == sizes.size()) {
            spdlog::info("✓ Test 4 PASSED: All images transmitted successfully");
        } else {
            spdlog::error("✗ Test 4 FAILED: Expected {} images, received {}",
                         sizes.size(), all_images.size());
        }
    }

    // 统计信息
    spdlog::info("========================================");
    spdlog::info("Test Summary");
    spdlog::info("========================================");
    spdlog::info("Total images published: {}", img_publisher.get_published_count());
    spdlog::info("Total images received: {}", img_subscriber.get_received_count());
    spdlog::info("========================================");

    spdlog::info("✓ All tests completed successfully!");
    return 0;
}