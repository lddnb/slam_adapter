#include "slam_common/iceoryx_pub_sub.hpp"
#include "slam_common/sensor_struct.hpp"
#include "slam_common/crash_logger.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <thread>

using namespace ms_slam::slam_common;

/**
 * @brief 构造模拟 IMU 数据
 * @param idx 序号
 * @return 模拟 IMU
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
 * @brief 构造模拟点云帧（仅填充少量点）
 * @param idx 帧序号
 * @return 模拟点云帧
 */
Mid360Frame MakeLidarSample(uint32_t idx)
{
    Mid360Frame frame{};
    frame.index = idx;
    frame.frame_timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    frame.frame_id.fill('\0');
    constexpr char kFrameId[] = "publisher_lidar";
    const std::size_t copy_len = std::min<std::size_t>(sizeof(kFrameId) - 1, frame.frame_id.size() - 1);
    std::copy_n(kFrameId, copy_len, frame.frame_id.begin());
    frame.point_count = 5;

    for (uint32_t i = 0; i < frame.point_count; ++i) {
        auto& p = frame.points[i];
        p.x = static_cast<float>(i) * 0.1F;
        p.y = static_cast<float>(i) * 0.2F;
        p.z = static_cast<float>(i) * 0.3F;
        p.intensity = static_cast<uint8_t>(i);
        p.tag = static_cast<uint8_t>(i % 4);
        p.timestamp_ns = frame.frame_timestamp_ns + i * 1000;
    }
    return frame;
}

int main()
{
    LoggerConfig config;
    config.log_file_path = "pub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(2));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("publisher_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("Failed to initialize crash logger");
        return 1;
    }

    auto node = std::make_shared<IoxNode>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create node"));

    IoxPublisher<Mid360Frame> lidar_pub(node, "/test/lidar");
    IoxPublisher<LivoxImuData> imu_pub(node, "/test/imu");

    for (uint32_t i = 0; i < 3; ++i) {
        auto lidar_sample = MakeLidarSample(i);
        auto imu_sample = MakeImuSample(i);

        // 按需绑定构建回调，直接将本次样本写入共享内存
        lidar_pub.SetBuildCallback([lidar_sample](Mid360Frame& payload) { payload = lidar_sample; });
        imu_pub.SetBuildCallback([imu_sample](LivoxImuData& payload) { payload = imu_sample; });

        lidar_pub.Publish();
        imu_pub.Publish();

        spdlog::info("Published lidar frame #{}, imu #{}, total lidar={}, imu={}",
                     i,
                     i,
                     lidar_pub.GetPublishedCount(),
                     imu_pub.GetPublishedCount());

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    spdlog::info("Publisher finished");
    return 0;
}
