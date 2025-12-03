#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <spdlog/spdlog.h>

#include "slam_common/callback_dispatcher.hpp"
#include "slam_common/iceoryx_pub_sub.hpp"
#include "slam_common/sensor_struct.hpp"

using namespace ms_slam::slam_common;

/**
 * @brief 构造模拟 IMU 数据
 * @param idx 序号
 * @return IMU 数据
 */
LivoxImuData MakeImuSample(uint32_t idx)
{
    LivoxImuData imu{};
    imu.timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    imu.index = idx;
    imu.angular_velocity = {0.1F * idx, 0.2F * idx, 0.3F * idx};
    imu.linear_acceleration = {1.0F * idx, 1.5F * idx, 2.0F * idx};
    return imu;
}

/**
 * @brief 构造模拟点云
 * @param idx 帧序号
 * @return 点云帧
 */
Mid360Frame MakeLidarSample(uint32_t idx)
{
    Mid360Frame frame{};
    frame.index = idx;
    frame.point_count = 3;
    frame.frame_timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    frame.frame_id.fill('\0');
    constexpr char kFrameId[] = "dispatcher_lidar";
    const std::size_t copy_len = std::min<std::size_t>(sizeof(kFrameId) - 1, frame.frame_id.size() - 1);
    std::copy_n(kFrameId, copy_len, frame.frame_id.begin());
    for (uint32_t i = 0; i < frame.point_count; ++i) {
        auto& p = frame.points[i];
        p.x = 0.1F * static_cast<float>(i);
        p.y = 0.2F * static_cast<float>(i);
        p.z = 0.3F * static_cast<float>(i);
        p.intensity = static_cast<uint8_t>(i);
        p.tag = static_cast<uint8_t>(i);
        p.timestamp_ns = frame.frame_timestamp_ns + i * 1000;
    }
    return frame;
}

int main()
{
    spdlog::info("=== CallbackDispatcher test (iceoryx2 fixed structs) ===");

    auto node = std::make_shared<IoxNode>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create node"));

    auto imu_pub = std::make_shared<IoxPublisher<LivoxImuData>>(node, "/dispatcher/imu");
    auto lidar_pub = std::make_shared<IoxPublisher<Mid360Frame>>(node, "/dispatcher/lidar");

    std::atomic<uint32_t> imu_count{0};
    std::atomic<uint32_t> lidar_count{0};

    auto imu_sub = std::make_shared<IoxSubscriber<LivoxImuData>>(
        node,
        "/dispatcher/imu",
        [&imu_count](const LivoxImuData&) { imu_count.fetch_add(1); });

    auto lidar_sub = std::make_shared<IoxSubscriber<Mid360Frame>>(
        node,
        "/dispatcher/lidar",
        [&lidar_count](const Mid360Frame&) { lidar_count.fetch_add(1); });

    CallbackDispatcher dispatcher;
    dispatcher.SetPollInterval(std::chrono::milliseconds(1));

    // 注册 IMU 回调
    dispatcher.RegisterSubscriber(imu_sub, "imu_sub", 10);
    // 注册 Lidar 回调
    dispatcher.RegisterSubscriber(lidar_sub, "lidar_sub", 5);

    // 发布线程
    std::thread pub_thread([&]() {
        for (uint32_t i = 0; i < 5; ++i) {
            auto imu_sample = MakeImuSample(i);
            imu_pub->SetBuildCallback([imu_sample](LivoxImuData& payload) { payload = imu_sample; });
            imu_pub->Publish();
            if (i % 2 == 0) {
                auto lidar_sample = MakeLidarSample(i);
                lidar_pub->SetBuildCallback([lidar_sample](Mid360Frame& payload) { payload = lidar_sample; });
                lidar_pub->Publish();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    dispatcher.Start();

    // 运行一段时间让回调处理发布的数据
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    dispatcher.Stop();
    dispatcher.PrintStatistics();

    pub_thread.join();

    spdlog::info("Received imu={}, lidar={}", imu_count.load(), lidar_count.load());
    spdlog::info("=== CallbackDispatcher test finished ===");
    return 0;
}
