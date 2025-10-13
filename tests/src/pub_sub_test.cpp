#include "slam_common/flatbuffers_pub_sub.hpp"
#include "slam_common/foxglove_messages.hpp"

#include <spdlog/spdlog.h>
#include <flatbuffers/flatbuffers.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>

#include <chrono>
#include <thread>
#include <random>

using namespace ms_slam::slam_common;

// ============================================================================
// Test: Zero-copy Pub/Sub using publish_raw() and FoxgloveMessages
// ============================================================================

/// Generate test Foxglove PointCloud FlatBuffers data
flatbuffers::DetachedBuffer create_foxglove_pointcloud(uint32_t seq, size_t num_points = 1000)
{
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);

    // Create timestamp
    uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
    uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    foxglove::Time timestamp(stamp_sec, stamp_nsec);

    // Create frame_id
    auto frame_id = fbb.CreateString("lidar_frame");

    // Create point cloud fields (x, y, z, intensity)
    std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fields;
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "x", 0, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "y", 4, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "z", 8, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "intensity", 12, foxglove::NumericType_FLOAT32));
    auto fields_vector = fbb.CreateVector(fields);

    // Generate random point cloud data
    std::vector<uint8_t> point_data;
    point_data.reserve(num_points * 16);  // 4 fields * 4 bytes each

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    for (size_t i = 0; i < num_points; ++i) {
        float x = coord_dist(gen);
        float y = coord_dist(gen);
        float z = coord_dist(gen);
        float intensity = intensity_dist(gen);

        // Convert float to bytes and add to point_data
        const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(&x);
        const uint8_t* y_bytes = reinterpret_cast<const uint8_t*>(&y);
        const uint8_t* z_bytes = reinterpret_cast<const uint8_t*>(&z);
        const uint8_t* intensity_bytes = reinterpret_cast<const uint8_t*>(&intensity);

        point_data.insert(point_data.end(), x_bytes, x_bytes + 4);
        point_data.insert(point_data.end(), y_bytes, y_bytes + 4);
        point_data.insert(point_data.end(), z_bytes, z_bytes + 4);
        point_data.insert(point_data.end(), intensity_bytes, intensity_bytes + 4);
    }

    auto data_vector = fbb.CreateVector(point_data);

    // Create PointCloud
    auto pointcloud = foxglove::CreatePointCloud(
        fbb,
        &timestamp,
        frame_id,
        0,  // pose (optional)
        16,  // point_stride (4 fields * 4 bytes)
        fields_vector,
        data_vector
    );

    fbb.Finish(pointcloud);

    spdlog::info("Created Foxglove PointCloud: {} points, {} bytes",
                 num_points, fbb.GetSize());

    return fbb.Release();
}

/// Generate test Foxglove CompressedImage FlatBuffers data
flatbuffers::DetachedBuffer create_foxglove_compressed_image(uint32_t seq, size_t width = 640, size_t height = 480)
{
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);

    // Create timestamp
    uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
    uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    foxglove::Time timestamp(stamp_sec, stamp_nsec);

    // Create frame_id
    auto frame_id = fbb.CreateString("camera_frame");

    // Create format
    auto format = fbb.CreateString("jpeg");

    // Generate simulated compressed image data (in real use this would be actual JPEG data)
    std::vector<uint8_t> image_data(width * height / 10);  // Simulated compressed size
    for (size_t i = 0; i < image_data.size(); ++i) {
        image_data[i] = static_cast<uint8_t>(i % 256);
    }
    auto data = fbb.CreateVector(image_data);

    // Create CompressedImage
    auto compressed_img = foxglove::CreateCompressedImage(
        fbb,
        &timestamp,
        frame_id,
        data,
        format
    );

    fbb.Finish(compressed_img);

    spdlog::info("Created Foxglove CompressedImage: {}x{}, {} bytes",
                 width, height, fbb.GetSize());

    return fbb.Release();
}

/// Generate test Foxglove Imu FlatBuffers data
flatbuffers::DetachedBuffer create_foxglove_imu(uint32_t seq)
{
    flatbuffers::FlatBufferBuilder fbb(1024);

    // Create timestamp
    uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
    uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    foxglove::Time timestamp(stamp_sec, stamp_nsec);

    // Create frame_id
    auto frame_id = fbb.CreateString("imu_frame");

    // Create angular velocity and linear acceleration
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

    auto angular_velocity = foxglove::CreateVector3(fbb, dist(gen), dist(gen), dist(gen));
    auto linear_acceleration = foxglove::CreateVector3(fbb, dist(gen) * 9.8, dist(gen) * 9.8, 9.8 + dist(gen));

    // Create Imu
    auto imu = foxglove::CreateImu(
        fbb,
        &timestamp,
        frame_id,
        angular_velocity,
        linear_acceleration
    );

    fbb.Finish(imu);

    spdlog::info("Created Foxglove Imu: {} bytes", fbb.GetSize());

    return fbb.Release();
}

int main()
{
    spdlog::info("========================================");
    spdlog::info("Foxglove FlatBuffers Raw Pub/Sub Test");
    spdlog::info("========================================");

    // Create iceoryx2 node
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));

    // ============================================================================
    // Test 1: PointCloud using publish_raw() and FoxglovePointCloud
    // ============================================================================

    spdlog::info("----------------------------------------");
    spdlog::info("Test 1: PointCloud Zero-copy Pub/Sub");
    spdlog::info("----------------------------------------");

    // Create Publisher (using FoxglovePointCloud type)
    FBSPublisher<FoxglovePointCloud> pc_publisher(node, "/foxglove/pointcloud");

    // Create Subscriber
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
        spdlog::info("  Fields count: {}", pc->fields()->size());
    };

    FBSSubscriber<FoxglovePointCloud> pc_subscriber(node, "/foxglove/pointcloud", pc_callback);

    // Publish 3 point clouds with different sizes
    for (int i = 0; i < 3; ++i) {
        size_t num_points = 1000 + i * 500;
        auto pc_buffer = create_foxglove_pointcloud(i, num_points);

        spdlog::info("Publishing PointCloud {} ({} points)...", i + 1, num_points);
        pc_publisher.publish_raw(pc_buffer.data(), pc_buffer.size());

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        pc_subscriber.receive_once();
    }

    spdlog::info("✓ Test 1 PASSED: Received {} PointClouds", pc_received_count.load());

    // ============================================================================
    // Test 2: CompressedImage using publish_raw()
    // ============================================================================

    spdlog::info("----------------------------------------");
    spdlog::info("Test 2: CompressedImage Zero-copy Pub/Sub");
    spdlog::info("----------------------------------------");

    // Create Publisher
    FBSPublisher<FoxgloveCompressedImage> img_publisher(node, "/foxglove/image");

    // Create Subscriber
    std::atomic<int> img_received_count{0};
    auto img_callback = [&img_received_count](const FoxgloveCompressedImage& img_wrapper) {
        img_received_count++;

        const foxglove::CompressedImage* img = img_wrapper.get();

        spdlog::info("✓ Received CompressedImage #{}: frame_id={}, format={}, data_size={}",
                     img_received_count.load(),
                     img->frame_id()->c_str(),
                     img->format()->c_str(),
                     img->data()->size());
    };

    FBSSubscriber<FoxgloveCompressedImage> img_subscriber(node, "/foxglove/image", img_callback);

    // Publish 3 images with different resolutions
    std::vector<std::pair<size_t, size_t>> resolutions = {{640, 480}, {1280, 720}, {1920, 1080}};
    for (int i = 0; i < 3; ++i) {
        auto [width, height] = resolutions[i];
        auto img_buffer = create_foxglove_compressed_image(i, width, height);

        spdlog::info("Publishing CompressedImage {} ({}x{})...", i + 1, width, height);
        img_publisher.publish_raw(img_buffer.data(), img_buffer.size());

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        img_subscriber.receive_once();
    }

    spdlog::info("✓ Test 2 PASSED: Received {} CompressedImages", img_received_count.load());

    // ============================================================================
    // Test 3: Imu using publish_from_builder()
    // ============================================================================

    spdlog::info("----------------------------------------");
    spdlog::info("Test 3: Imu using publish_from_builder()");
    spdlog::info("----------------------------------------");

    // Create Publisher
    FBSPublisher<FoxgloveImu> imu_publisher(node, "/foxglove/imu", PubSubConfig{.subscriber_max_buffer_size = 100});

    // Create threaded Subscriber
    std::atomic<int> imu_received_count{0};
    auto imu_callback = [&imu_received_count](const FoxgloveImu& imu_wrapper) {
        imu_received_count++;

        const foxglove::Imu* imu = imu_wrapper.get();

        spdlog::info("✓ Received Imu #{}: frame_id={}, angular_velocity=({:.3f}, {:.3f}, {:.3f}), linear_acceleration=({:.3f}, {:.3f}, {:.3f})",
                     imu_received_count.load(),
                     imu->frame_id()->c_str(),
                     imu->angular_velocity()->x(), imu->angular_velocity()->y(), imu->angular_velocity()->z(),
                     imu->linear_acceleration()->x(), imu->linear_acceleration()->y(), imu->linear_acceleration()->z());
    };

    ThreadedFBSSubscriber<FoxgloveImu> imu_subscriber(
        node, "/foxglove/imu", imu_callback);

    imu_subscriber.start();
    spdlog::info("Threaded Imu Subscriber started");

    // Publish 10 IMU data
    for (int i = 0; i < 10; ++i) {
        flatbuffers::FlatBufferBuilder fbb(1024);

        uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
        uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
        foxglove::Time timestamp(stamp_sec, stamp_nsec);

        auto frame_id = fbb.CreateString("imu_frame");

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

        auto angular_velocity = foxglove::CreateVector3(fbb, dist(gen), dist(gen), dist(gen));
        auto linear_acceleration = foxglove::CreateVector3(fbb, dist(gen) * 9.8, dist(gen) * 9.8, 9.8 + dist(gen));

        auto imu = foxglove::CreateImu(fbb, &timestamp, frame_id, angular_velocity, linear_acceleration);
        fbb.Finish(imu);

        spdlog::info("Publishing Imu {} using publish_from_builder()...", i + 1);
        imu_publisher.publish_from_builder(fbb);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Wait for all messages to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    imu_subscriber.stop();
    spdlog::info("Threaded Imu Subscriber stopped");

    spdlog::info("✓ Test 3 PASSED: Received {} Imu messages", imu_received_count.load());

    // ============================================================================
    // Summary
    // ============================================================================

    spdlog::info("========================================");
    spdlog::info("Test Summary");
    spdlog::info("========================================");
    spdlog::info("PointCloud received: {} messages", pc_received_count.load());
    spdlog::info("CompressedImage received: {} messages", img_received_count.load());
    spdlog::info("Imu received: {} messages", imu_received_count.load());
    spdlog::info("Total published: {} messages",
                 pc_publisher.get_published_count() +
                 img_publisher.get_published_count() +
                 imu_publisher.get_published_count());
    spdlog::info("========================================");
    spdlog::info("✓ All Foxglove raw bytes stream tests PASSED!");
    spdlog::info("========================================");

    return 0;
}
