// Copyright (c) 2024 Contributors to the Eclipse Foundation
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

#include "generic_publisher_subscriber.hpp"
#include "data_types.hpp"

#include <iostream>
#include <random>
#include <csignal>

#include <spdlog/spdlog.h>
#include <cpptrace/cpptrace.hpp>

// 自定义信号处理函数
void signal_handler(int signal_number) {
    // 使用 cpptrace 获取并打印堆栈跟踪
    cpptrace::generate_trace().print();
    
    // 退出程序
    std::_Exit(EXIT_FAILURE);
}

// Generator function for PointCloud messages
PointCloud generate_point_cloud(uint32_t seq) {
    PointCloud cloud;
    cloud.seq = seq;
    cloud.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Generate some random points
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    cloud.points.clear();
    uint32_t num_points = 10 + (seq % 100); // 100-200 points
    cloud.num_points = num_points;
    std::cout << "num_points: " << num_points << std::endl;

    for (uint32_t i = 0; i < num_points; ++i) {
        PointCloudPoint point;
        point.tov = cloud.timestamp;
        point.x = coord_dist(gen);
        point.y = coord_dist(gen);
        point.z = coord_dist(gen);
        point.intensity = intensity_dist(gen);
        point.return_order = i % 3; // 0, 1, or 2
        cloud.points.push_back(point);
    }

    return cloud;
}

// Generator function for Image messages
Image generate_image(uint32_t seq) {
    Image image;
    image.seq = seq;
    image.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Set image format
    image.format.width = 640;
    image.format.height = 480;
    image.format.stride = image.format.width * 3; // RGB
    image.format.pixel_format[0] = 'R';
    image.format.pixel_format[1] = 'G';
    image.format.pixel_format[2] = 'B';
    image.format.pixel_format[3] = '8';

    // Generate simple pattern
    image.data.clear();
    for (uint32_t y = 0; y < image.format.height; ++y) {
        for (uint32_t x = 0; x < image.format.width; ++x) {
            uint8_t r = static_cast<uint8_t>((x + seq) % 256);
            uint8_t g = static_cast<uint8_t>((y + seq) % 256);
            uint8_t b = static_cast<uint8_t>((x + y + seq) % 256);

            image.data.push_back(r);
            image.data.push_back(g);
            image.data.push_back(b);
        }
    }

    return image;
}

int main() {
    using namespace iox2;

    // 注册信号处理函数，捕获常见的会导致程序崩溃的信号
    std::signal(SIGSEGV, signal_handler); // 非法内存访问
    std::signal(SIGABRT, signal_handler); // 中止信号（例如 assert 失败或调用 abort()）
    std::signal(SIGFPE, signal_handler);  // 浮点异常
    std::signal(SIGILL, signal_handler);  // 非法指令

    cpptrace::register_terminate_handler();
    // cpptrace::register_signal_handler();

    // Initialize node
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        NodeBuilder().create<ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    // Create PointCloud publisher with generator function
    PointCloudPublisher pc_publisher(
        node,
        "Generic/PointCloud/Service",
        generate_point_cloud,
        std::chrono::milliseconds(500) // Publish every 500ms
    );

    // Create PointCloud subscriber with callback
    PointCloudSubscriber pc_subscriber(
        node,
        "Generic/PointCloud/Service",
        [](const PointCloud& cloud) {
            std::cout << "Received PointCloud: seq=" << cloud.seq
                      << ", points=" << cloud.points.size() << std::endl;
        }
    );

    // Create Image publisher with generator function
    ImagePublisher img_publisher(
        node,
        "Generic/Image/Service",
        generate_image,
        std::chrono::milliseconds(1000) // Publish every 1000ms
    );

    // Create Image subscriber with callback
    ImageSubscriber img_subscriber(
        node,
        "Generic/Image/Service",
        [](const Image& image) {
            std::cout << "Received Image: seq=" << image.seq
                      << ", size=" << image.format.width << "x" << image.format.height
                      << ", data_size=" << image.data.size() << std::endl;
        }
    );

    // Set additional callbacks for monitoring
    pc_publisher.set_publish_callback([](uint32_t seq, const PointCloud& cloud) {
        std::cout << "Published PointCloud " << seq << " with " << cloud.points.size() << " points" << std::endl;
    });

    img_publisher.set_publish_callback([](uint32_t seq, const Image& image) {
        std::cout << "Published Image " << seq << " (" << image.format.width << "x" << image.format.height << ")" << std::endl;
    });

    // Start all components
    std::cout << "Starting publishers and subscribers..." << std::endl;
    pc_publisher.start();
    pc_subscriber.start();
    img_publisher.start();
    img_subscriber.start();

    // Run for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Stop all components
    std::cout << "Stopping all components..." << std::endl;
    pc_publisher.stop();
    pc_subscriber.stop();
    img_publisher.stop();
    img_subscriber.stop();

    // Print statistics
    std::cout << "Statistics:" << std::endl;
    std::cout << "  PointCloud: Published=" << pc_publisher.get_published_count()
              << ", Received=" << pc_subscriber.get_received_count() << std::endl;
    std::cout << "  Image: Published=" << img_publisher.get_published_count()
              << ", Received=" << img_subscriber.get_received_count() << std::endl;

    // 下面是一个故意制造崩溃的例子
    int* ptr = nullptr;
    *ptr = 42; // 这将会触发 SIGSEGV

    // Demonstrate single-shot publishing/receiving
    std::cout << "\nDemonstrating single-shot operations..." << std::endl;

    // Create single message and publish
    auto single_cloud = generate_point_cloud(10);
    std::cout << "single_cloud: " << single_cloud.num_points << std::endl;
    bool success = pc_publisher.publish(single_cloud);
    std::cout << "Single publish result: " << (success ? "success" : "failed") << std::endl;

    // Try to receive single message
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto received_cloud = pc_subscriber.receive_once();
    if (received_cloud.has_value()) {
        std::cout << "Single receive: got cloud with " << received_cloud->points.size() << " points" << std::endl;
    } else {
        std::cout << "Single receive: no message available" << std::endl;
    }

    std::cout << "Demo completed!" << std::endl;
    return 0;
}