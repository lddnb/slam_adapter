#include "slam_common/iceoryx_pub_sub.hpp"
#include "slam_common/sensor_struct.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace ms_slam::slam_common;

/**
 * @brief 构造随机 IMU 数据
 * @param idx 序号
 * @return IMU 数据
 */
LivoxImuData MakeImuSample(uint32_t idx)
{
    LivoxImuData imu{};
    imu.timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    imu.index = idx;

    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    for (int i = 0; i < 3; ++i) {
        imu.angular_velocity[i] = dist(gen);
        imu.linear_acceleration[i] = dist(gen) * 9.8F;
    }
    return imu;
}

/**
 * @brief 构造随机点云帧
 * @param idx 帧序号
 * @param point_count 点数
 * @return 点云帧
 */
LivoxPointCloudDate MakeLidarSample(uint32_t idx, uint32_t point_count)
{
    LivoxPointCloudDate frame{};
    frame.index = idx;
    frame.frame_timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    frame.frame_id.fill('\0');
    constexpr char kFrameId[] = "test_lidar";
    const std::size_t copy_len = std::min<std::size_t>(sizeof(kFrameId) - 1, frame.frame_id.size() - 1);
    std::copy_n(kFrameId, copy_len, frame.frame_id.begin());
    frame.point_count = std::min(point_count, static_cast<uint32_t>(kLivoxMaxPoints));

    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(-5.0F, 5.0F);

    for (uint32_t i = 0; i < frame.point_count; ++i) {
        auto& p = frame.points[i];
        p.x = dist(gen);
        p.y = dist(gen);
        p.z = dist(gen);
        p.intensity = static_cast<uint8_t>(i & 0xFF);
        p.tag = static_cast<uint8_t>(i % 4);
        p.timestamp_ns = frame.frame_timestamp_ns + i * 1000;
    }
    return frame;
}

int main()
{
    spdlog::info("========================================");
    spdlog::info("iceoryx2 fixed-struct Pub/Sub Test");
    spdlog::info("========================================");

    auto node = std::make_shared<IoxNode>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create node"));

    IoxPublisher<LivoxPointCloudDate> pc_pub(node, "/test/lidar");
    IoxSubscriber<LivoxPointCloudDate> pc_sub(node, "/test/lidar");

    IoxPublisher<LivoxImuData> imu_pub(node, "/test/imu");
    IoxSubscriber<LivoxImuData> imu_sub(node, "/test/imu");

    std::atomic<uint32_t> pc_recv{0};
    std::atomic<uint32_t> imu_recv{0};

    pc_sub.SetReceiveCallback([&pc_recv](const LivoxPointCloudDate& msg) {
        ++pc_recv;
        spdlog::info("Received lidar frame #{} with {} points", msg.index, msg.point_count);
    });
    imu_sub.SetReceiveCallback([&imu_recv](const LivoxImuData& msg) {
        ++imu_recv;
        spdlog::info("Received imu #{} at {}", msg.index, msg.timestamp_ns);
    });

    for (uint32_t i = 0; i < 3; ++i) {
        auto frame = MakeLidarSample(i, 8 + i * 2);
        pc_pub.SetBuildCallback([frame](LivoxPointCloudDate& payload) { payload = frame; });
        pc_pub.Publish();

        auto imu = MakeImuSample(i);
        imu_pub.SetBuildCallback([imu](LivoxImuData& payload) { payload = imu; });
        imu_pub.Publish();

        // 收取
        pc_sub.ReceiveAll();
        imu_sub.ReceiveAll();

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 图像相关测试暂时禁用，等待定长图像结构定义
    // TODO: add image publish/subscribe when a fixed-size image struct is available.

    spdlog::info("Summary: lidar received {}, imu received {}", pc_recv.load(), imu_recv.load());
    spdlog::info("========================================");
    spdlog::info("Test finished");
    return 0;
}
