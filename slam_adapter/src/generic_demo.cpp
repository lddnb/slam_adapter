#include "generic_publisher_subscriber.hpp"
#include "data_types.hpp"
#include "slam_common/signal_handler.hpp"
#include "slam_concepts.hpp"

#include <sys/wait.h>
#include <iostream>
#include <random>
#include <csignal>
#include <ranges>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <cpptrace/cpptrace.hpp>
#include <fbs/monster_generated.h>

// C++20 Optimized point cloud generator with concepts
template <slam_concepts::BasicPoint PointType>
PointCloud<PointType> generate_point_cloud(uint32_t seq)
{
    PointCloud<PointType> cloud;
    cloud.seq = seq;
    cloud.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // C++20 constexpr random generation setup
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    cloud.clear();  // Use our new optimized clear method
    uint32_t num_points = 10 + (seq % 100);

    // C++20 ranges-based generation
    auto point_indices = std::views::iota(0u, num_points);

    for ([[maybe_unused]] auto i : point_indices) {
        PointType point{};

        // Use our optimized helper functions
        slam_generators::set_basic_fields(point, coord_dist(gen), coord_dist(gen), coord_dist(gen));

        // C++20 if constexpr with concepts - more readable than requires
        if constexpr (slam_concepts::HasTimestamp<PointType>) {
            slam_generators::set_timestamp(point, cloud.timestamp);
        }
        if constexpr (slam_concepts::HasIntensity<PointType>) {
            slam_generators::set_intensity(point, intensity_dist(gen));
        }
        if constexpr (slam_concepts::HasColor<PointType>) {
            slam_generators::set_color(
                point,
                static_cast<uint8_t>(gen() % 256),
                static_cast<uint8_t>(gen() % 256),
                static_cast<uint8_t>(gen() % 256));
        }

        cloud.emplace_point(std::move(point));  // Use perfect forwarding
    }

    return cloud;
}

// Generator function for Image messages
Image generate_image(uint32_t seq)
{
    Image image;
    image.seq = seq;
    image.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // Set image format
    image.format.width = 640;
    image.format.height = 480;
    image.format.stride = image.format.width * 3;  // RGB
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

int main()
{
    using namespace iox2;

    ms_slam::slam_common::warmup_cpptrace();
    ms_slam::slam_common::InstallFailureSignalHandler();

    // Initialize node
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(NodeBuilder().create<ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    // Create PointCloud publisher with generator function
    PointITCloudPublisher pc_publisher(
        node,
        "Generic/PointCloud/Service",
        generate_point_cloud<PointIT>,
        {.publish_interval = std::chrono::milliseconds(500)}  // C++20 designated initializer
    );

    // Create PointCloud subscriber with callback
    PointITCloudSubscriber pc_subscriber(node, "Generic/PointCloud/Service", [](const PointCloud<PointIT>& cloud) {
        std::cout << "Received PointCloud: seq=" << cloud.seq << ", points=" << cloud.points.size() << std::endl;
    });

    // Create Image publisher with generator function
    ImagePublisher img_publisher(
        node,
        "Generic/Image/Service",
        generate_image,
        {.publish_interval = std::chrono::milliseconds(1000)}  // C++20 designated initializer
    );

    // Create Image subscriber with callback
    ImageSubscriber img_subscriber(node, "Generic/Image/Service", [](const Image& image) {
        std::cout << "Received Image: seq=" << image.seq << ", size=" << image.format.width << "x" << image.format.height
                  << ", data_size=" << image.data.size() << std::endl;
    });

    // Set additional callbacks for monitoring
    pc_publisher.set_publish_callback([](uint32_t seq, const PointCloud<PointIT>& cloud) {
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
    std::cout << "  PointCloud: Published=" << pc_publisher.get_published_count() << ", Received=" << pc_subscriber.get_received_count() << std::endl;
    std::cout << "  Image: Published=" << img_publisher.get_published_count() << ", Received=" << img_subscriber.get_received_count() << std::endl;

    // 下面是一个故意制造崩溃的例子
    // int* ptr = nullptr;
    // *ptr = 42; // 这将会触发 SIGSEGV

    std::cout << "Demo completed!" << std::endl;
    return 0;
}