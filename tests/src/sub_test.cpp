#include "slam_common/iceoryx_pub_sub.hpp"
#include "slam_common/sensor_struct.hpp"
#include "slam_common/crash_logger.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>
#include <thread>

using namespace ms_slam::slam_common;

/**
 * @brief 订阅回调，打印 IMU 数据
 * @param imu IMU 数据引用
 */
void OnImu(const LivoxImuData& imu)
{
    spdlog::info(
        "IMU idx={}, ts={}, gyro=({:.3f},{:.3f},{:.3f}), acc=({:.3f},{:.3f},{:.3f})",
        imu.index,
        imu.timestamp_ns,
        imu.angular_velocity[0],
        imu.angular_velocity[1],
        imu.angular_velocity[2],
        imu.linear_acceleration[0],
        imu.linear_acceleration[1],
        imu.linear_acceleration[2]);
}

/**
 * @brief 订阅回调，打印点云基本信息
 * @param frame 点云帧引用
 */
void OnLidar(const LivoxPointCloudDate& frame)
{
    spdlog::info("Lidar frame idx={}, ts={}, points={}", frame.index, frame.frame_timestamp_ns, frame.point_count);
    if (frame.point_count > 0) {
        const auto& p = frame.points[0];
        spdlog::info("  first point: ({:.3f},{:.3f},{:.3f}) id={} tag={} ts={}", p.x, p.y, p.z, p.intensity, p.tag, p.timestamp_ns);
    }
}

int main()
{
    LoggerConfig config;
    config.log_file_path = "sub.log";
    auto dup_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(2));
    dup_filter->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    dup_filter->add_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(config.log_file_path, true));
    auto logger = std::make_shared<spdlog::logger>("subscriber_logger", dup_filter);

    logger->set_pattern(config.log_pattern);
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);

    if (!SLAM_CRASH_LOGGER_INIT(logger)) {
        spdlog::error("Failed to initialize crash logger");
        return 1;
    }

    auto node = std::make_shared<IoxNode>(iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Create node"));

    IoxSubscriber<LivoxPointCloudDate> lidar_sub(node, "/test/lidar", OnLidar);
    IoxSubscriber<LivoxImuData> imu_sub(node, "/test/imu", OnImu);

    // 简单轮询等待外部 publisher 推送
    for (int i = 0; i < 20; ++i) {
        imu_sub.ReceiveAll();
        lidar_sub.ReceiveAll();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    spdlog::info("Subscriber finished");
    return 0;
}
