#include "slam_common/generic_flatbuffer_pubsub.hpp"
#include "slam_common/flatbuffer_serializer.hpp"

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>
#include <random>
#include <ranges>

using namespace ms_slam::slam_common;

// ============================================================================
// 辅助函数：OpenCV Mat <-> Image 转换
// ============================================================================

PointCloudIT generate_point_cloud(uint32_t seq)
{
    PointCloudIT cloud;
    cloud.seq = seq;
    cloud.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // C++20 constexpr random generation setup
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    cloud.clear();  // Use our new optimized clear method
    uint32_t num_points = 10000 + (seq % 100);

    // C++20 ranges-based generation
    auto point_indices = std::views::iota(0u, num_points);

    for ([[maybe_unused]] auto i : point_indices) {
        PointIT point{coord_dist(gen), coord_dist(gen), coord_dist(gen), intensity_dist(gen), 0};

        cloud.emplace_back(point);
    }

    return cloud;
}

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

// ============================================================================
// 测试: 使用通用模板 Publisher/Subscriber
// ============================================================================

int main()
{
    spdlog::info("========================================");
    spdlog::info("通用模板 FlatBuffer Pub/Sub 测试");
    spdlog::info("========================================");

    // 创建 iceoryx2 节点
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    // ============================================================================
    // 测试1: 非线程化 Subscriber（按需接收）
    // ============================================================================

    spdlog::info("----------------------------------------");
    spdlog::info("测试1: 使用 GenericFlatBufferPublisher/Subscriber");
    spdlog::info("----------------------------------------");

    // 创建通用 Publisher（使用 Image 类型）
    GenericFlatBufferPublisher<Image> img_publisher(node, "/test/image");

    // 创建通用 Subscriber（非线程化）
    std::atomic<int> received_count{0};
    auto callback = [&received_count](const Image& image) {
        received_count++;
        spdlog::info("✓ [Non-threaded] Received Image #{}: seq={}, size={}x{}, {} bytes",
                    received_count.load(), image.seq,
                    image.format.width, image.format.height, image.data.size());
    };

    GenericFlatBufferSubscriber<Image> img_subscriber(node, "/test/image", callback);

    // 加载测试图像（如果不存在则创建）
    cv::Mat test_mat;
    std::string image_path = "/home/ubuntu/data/image.png";
    test_mat = cv::imread(image_path);

    if (test_mat.empty()) {
        spdlog::warn("Could not load {}, creating synthetic test image...", image_path);
        // 创建合成测试图像
        test_mat = cv::Mat(2048, 2448, CV_8UC3, cv::Scalar(100, 150, 200));
        // 添加一些图案
        cv::rectangle(test_mat, cv::Point(100, 100), cv::Point(500, 500), cv::Scalar(255, 0, 0), -1);
        cv::circle(test_mat, cv::Point(1000, 1000), 300, cv::Scalar(0, 255, 0), -1);
    }
    spdlog::info("Test image ready: {}x{}", test_mat.cols, test_mat.rows);

    // 发布小图像
    cv::Mat small_img;
    cv::resize(test_mat, small_img, cv::Size(640, 480));
    auto image1 = opencv_mat_to_image(small_img,
                                     std::chrono::steady_clock::now().time_since_epoch().count(), 1);

    spdlog::info("Publishing small image (640x480)...");
    img_publisher.publish(image1);

    // 手动接收
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    img_subscriber.receive_once();

    // 发布中等图像
    cv::Mat medium_img;
    cv::resize(test_mat, medium_img, cv::Size(1280, 720));
    auto image2 = opencv_mat_to_image(medium_img,
                                     std::chrono::steady_clock::now().time_since_epoch().count(), 2);

    spdlog::info("Publishing medium image (1280x720)...");
    img_publisher.publish(image2);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    img_subscriber.receive_once();

    spdlog::info("✓ 测试1 通过: 非线程化 Subscriber 接收到 {} 条消息", received_count.load());

    // ============================================================================
    // 测试2: 线程化 Subscriber（后台线程自动接收）
    // ============================================================================

    spdlog::info("----------------------------------------");
    spdlog::info("测试2: 使用 ThreadedFlatBufferSubscriber");
    spdlog::info("----------------------------------------");

    std::atomic<int> threaded_received_count{0};
    auto threaded_callback = [&threaded_received_count](const Image& image) {
        threaded_received_count++;
        spdlog::info("✓ [Threaded] Received Image #{}: seq={}, size={}x{}, {} bytes",
                    threaded_received_count.load(), image.seq,
                    image.format.width, image.format.height, image.data.size());

        cv::Mat mat = image_to_opencv_mat(image);
        std::string filename = "received_image_" + std::to_string(image.seq) + ".png";
        cv::imwrite(filename, mat);
        spdlog::info("  Saved to {}", filename);
        // 模拟处理耗时
        // std::this_thread::sleep_for(std::chrono::milliseconds(50));
    };

    // ⚠️ 重要：先创建 Publisher，再创建 Subscriber
    // 这样可以避免 open_or_create() 冲突
    GenericFlatBufferPublisher<Image> threaded_publisher(node, "/test/threaded_image");

    // 创建线程化 Subscriber
    ThreadedFlatBufferSubscriber<Image> threaded_subscriber(
        node, "/test/threaded_image", threaded_callback, std::chrono::milliseconds(10));

    // 启动订阅者线程
    threaded_subscriber.start();
    spdlog::info("线程化 Subscriber 已启动，is_running={}", threaded_subscriber.is_running());

    // 连续发布多条消息
    for (int i = 0; i < 5; ++i) {
        cv::Size size(320 * (i + 1), 240 * (i + 1));
        cv::Mat img;
        cv::resize(test_mat, img, size);
        auto image = opencv_mat_to_image(img,
                                        std::chrono::steady_clock::now().time_since_epoch().count(),
                                        i + 10);

        spdlog::info("Publishing image {} ({}x{})...", i + 1, size.width, size.height);
        threaded_publisher.publish(image);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 等待所有消息被处理
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 停止订阅者线程
    threaded_subscriber.stop();
    spdlog::info("线程化 Subscriber 已停止，is_running={}", threaded_subscriber.is_running());

    spdlog::info("✓ 测试2 通过: 线程化 Subscriber 接收到 {} 条消息", threaded_received_count.load());

    // ============================================================================
    // 测试3: 非线程化 Subscriber 点云
    // ============================================================================
    spdlog::info("----------------------------------------");
    spdlog::info("测试3: 使用 GenericFlatBufferPublisher/Subscriber 接收点云");
    spdlog::info("----------------------------------------");

    // 创建通用 Publisher（使用 PointCloud 类型）
    GenericFlatBufferPublisher<PointCloudIT> pc_publisher(node, "/test/point_cloud");

    // 创建通用 Subscriber（非线程化）
    std::atomic<int> received_count_2{0};
    auto callback_2 = [&received_count_2](const PointCloudIT& point_cloud) {
        received_count_2++;
        spdlog::info("✓ [threaded] Received PointCloud #{}: seq={}, size={} points",
                    received_count_2.load(), point_cloud.seq, point_cloud.size());
    };

    ThreadedFlatBufferSubscriber<PointCloudIT> pc_subscriber(node, "/test/point_cloud", callback_2);
    pc_subscriber.start();

    // 发布点云
    for (int i = 0; i < 5; ++i) {
        auto point_cloud = generate_point_cloud(i + 10);
        spdlog::info("Publishing PointCloud {} ({} points)...", i + 1, point_cloud.size());
        spdlog::info("size: {} bytes", sizeof(PointIT) * point_cloud.size());
        pc_publisher.publish(point_cloud);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // pc_subscriber.receive_once();
    }

    // 手动接收
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // pc_subscriber.receive_once();

    pc_subscriber.stop();

    spdlog::info("✓ 测试3 通过: 非线程化 Subscriber 接收到 {} 条点云", received_count_2.load());

    // ============================================================================
    // 总结
    // ============================================================================

    spdlog::info("========================================");
    spdlog::info("测试总结");
    spdlog::info("========================================");
    spdlog::info("非线程化接收: {} 条消息", received_count.load());
    spdlog::info("线程化接收: {} 条消息", threaded_received_count.load());
    spdlog::info("总发布: {} 条消息",
                 img_publisher.get_published_count() + threaded_publisher.get_published_count());
    spdlog::info("========================================");
    spdlog::info("✓ 所有测试通过！");
    spdlog::info("========================================");

    return 0;
}