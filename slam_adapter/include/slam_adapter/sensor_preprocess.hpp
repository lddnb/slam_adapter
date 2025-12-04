#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

#include <slam_common/sensor_struct.hpp>
#include <slam_core/image.hpp>
#include <slam_core/imu.hpp>
#include <slam_core/point_cloud.hpp>
#include <slam_core/point_types.hpp>
#ifdef USE_PCL
#include <slam_core/PCL.hpp>
#endif

namespace ms_slam::slam_adapter
{
namespace
{
/// @brief 纳秒转秒的转换系数
inline constexpr double kNsToSeconds = 1e-9;

/**
 * @brief 将纳秒时间戳转换为秒
 * @param timestamp_ns 输入的纳秒时间戳
 * @return 秒
 */
inline double ToSeconds(uint64_t timestamp_ns) noexcept
{
    // 拆分秒与纳秒再合成，避免大整数直接转 double 造成的精度损失
    const uint64_t sec = timestamp_ns / 1000000000ULL;
    const uint64_t nsec = timestamp_ns % 1000000000ULL;
    return static_cast<double>(sec) + static_cast<double>(nsec) * kNsToSeconds;
}

/// @brief 记录上一帧末尾点的时间戳（秒），保证跨帧时间单调
inline double g_last_tail_timestamp = -std::numeric_limits<double>::infinity();
}  // namespace

/**
 * @brief 将 Livox 点云帧转换为 slam_core 点云
 * @param frame 输入的 Livox 点云帧
 * @param cloud 目标点云容器
 * @param blind_dist 盲区半径（米）
 * @return 转换成功返回 true
 */
inline bool ConvertLivoxPointCloudDate(
    const slam_common::LivoxPointCloudDate& frame,
    const std::shared_ptr<slam_core::PointCloud<slam_core::PointXYZITDescriptor>>& cloud,
    double blind_dist = 0.5)
{
    if (!cloud) {
        spdlog::warn("ConvertLivoxPointCloudMessage: cloud pointer is null");
        return false;
    }

    const uint32_t count = std::min<uint32_t>(frame.point_count, slam_common::kLivoxMaxPoints);
    if (count == 0) {
        cloud->clear();
        return false;
    }

    const double blind_sq = blind_dist * blind_dist;
    cloud->clear();
    cloud->reserve(count);

    const double prev_tail = g_last_tail_timestamp;
    double current_tail = -std::numeric_limits<double>::infinity();

    for (uint32_t i = 0; i < count; ++i) {
        const auto& p = frame.points[i];
        const double norm_sq = p.x * p.x + p.y * p.y + p.z * p.z;
        const double timestamp = ToSeconds(p.timestamp_ns);

        // 过滤掉盲区点及时间戳回退的点，确保下一帧的点时间始终大于上一帧尾部
        if (!std::isfinite(norm_sq) || norm_sq < blind_sq || timestamp <= prev_tail) {
            continue;
        }

        current_tail = std::max(current_tail, timestamp);
        cloud->push_back(slam_core::PointXYZIT(p.x, p.y, p.z, static_cast<float>(p.intensity), timestamp));
    }

    cloud->sort();
    if (!cloud->empty()) {
        g_last_tail_timestamp = current_tail;
        return true;
    }
    return false;
}

/**
 * @brief 将 Livox IMU 数据转换为 slam_core IMU
 * @param imu_in 输入的 LivoxImuData
 * @param imu_out 输出的 slam_core::IMU
 * @return 转换成功返回 true
 */
inline bool ConvertLivoxImuData(const slam_common::LivoxImuData& imu_in, slam_core::IMU& imu_out)
{
    Eigen::Vector3d gyro(imu_in.angular_velocity[0], imu_in.angular_velocity[1], imu_in.angular_velocity[2]);
    Eigen::Vector3d accel(imu_in.linear_acceleration[0], imu_in.linear_acceleration[1], imu_in.linear_acceleration[2]);

    const double timestamp = ToSeconds(imu_in.timestamp_ns);
    if (!std::isfinite(timestamp)) {
        spdlog::warn("ConvertLivoxImuData: invalid timestamp {}", imu_in.timestamp_ns);
        return false;
    }
    const uint64_t index = imu_in.index;

    imu_out = slam_core::IMU(gyro, accel, timestamp, index);
    return true;
}

/**
 * @brief 解码定长图像消息为 slam_core 图像
 * @tparam ImageStruct 支持的图像结构
 * @param message 输入的定长图像
 * @param image_out 输出的 slam_core::Image
 * @return 解码成功返回 true
 */
inline bool DecodeImageMessage(const slam_common::ImageDate& message, slam_core::Image& image_out)
{
    const auto& header = message.header;
    if (header.compressed) {
        spdlog::warn("DecodeImageMessage: compressed flag is true, skipping");
        image_out = slam_core::Image();
        return false;
    }

    const int width = static_cast<int>(header.width);
    const int height = static_cast<int>(header.height);
    if (width <= 0 || height <= 0) {
        spdlog::warn("DecodeImageMessage: invalid image size {}x{}", width, height);
        image_out = slam_core::Image();
        return false;
    }

    const std::size_t expected_size = static_cast<std::size_t>(header.step) * static_cast<std::size_t>(height);
    if (expected_size == 0 || expected_size > message.data.size()) {
        spdlog::warn("DecodeImageMessage: payload size mismatch, expected {}, buffer {}", expected_size, message.data.size());
        image_out = slam_core::Image();
        return false;
    }

    // 仅支持 3 通道 8bit 图像，其他编码暂不处理
    cv::Mat mat(height, width, CV_8UC3);
    std::memcpy(mat.data, message.data.data(), expected_size);

    image_out = slam_core::Image(mat, ToSeconds(header.timestamp_ns));
    return true;
}
}  // namespace ms_slam::slam_adapter
