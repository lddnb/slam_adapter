#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <slam_common/foxglove_messages.hpp>
#include <slam_core/image.hpp>
#include <slam_core/imu.hpp>
#include <slam_core/point_cloud.hpp>
#include <slam_core/point_types.hpp>

namespace ms_slam::slam_adapter
{
inline bool ConvertHesaiPointCloudMessage(
    const foxglove::PointCloud& message,
    const std::shared_ptr<slam_core::PointCloud<slam_core::PointXYZITDescriptor>>& cloud)
{
    if (!cloud) {
        return false;
    }

    const auto data_buffer = message.data();
    const std::uint32_t stride = message.point_stride();
    if (!data_buffer || stride == 0) {
        cloud->clear();
        return false;
    }

    constexpr std::uint32_t x_offset = 0;
    constexpr std::uint32_t y_offset = 4;
    constexpr std::uint32_t z_offset = 8;
    constexpr std::uint32_t intensity_offset = 12;
    constexpr std::uint32_t timestamp_offset = 18;

    if (stride < timestamp_offset + sizeof(double)) {
        cloud->clear();
        return false;
    }

    const std::size_t data_size = data_buffer->size();
    if (data_size % message.point_stride() != 0) {
        cloud->clear();
        return false;
    }

    const std::size_t num_points = data_size / message.point_stride();
    cloud->clear();
    cloud->reserve(num_points);

    for (std::size_t i = 0; i < num_points; ++i) {
        const auto* data = data_buffer->Data() + i * stride;

        const float x = *reinterpret_cast<const float*>(data + x_offset);
        const float y = *reinterpret_cast<const float*>(data + y_offset);
        const float z = *reinterpret_cast<const float*>(data + z_offset);
        const float intensity = *reinterpret_cast<const float*>(data + intensity_offset);
        const double timestamp = *reinterpret_cast<const double*>(data + timestamp_offset);

        cloud->push_back(slam_core::PointXYZIT(x, y, z, intensity, timestamp));
    }

    return true;
}

inline bool ConvertLivoxPointCloudMessage(
    const foxglove::PointCloud& message,
    const std::shared_ptr<slam_core::PointCloud<slam_core::PointXYZITDescriptor>>& cloud)
{
    if (!cloud) {
        return false;
    }

    const auto data_buffer = message.data();
    const std::uint32_t stride = message.point_stride();
    const auto* stamp = message.timestamp();
    if (!data_buffer || stride == 0 ||!stamp) {
        cloud->clear();
        return false;
    }

    constexpr std::uint32_t x_offset = 0;
    constexpr std::uint32_t y_offset = 4;
    constexpr std::uint32_t z_offset = 8;
    constexpr std::uint32_t reflectivity_offset = 12;
    constexpr std::uint32_t tag_offset = 13;
    constexpr std::uint32_t line_offset = 14;
    constexpr std::uint32_t timestamp_offset = 16;

    if (stride < timestamp_offset + sizeof(uint32_t)) {
        cloud->clear();
        return false;
    }

    const std::size_t data_size = data_buffer->size();
    if (data_size % message.point_stride() != 0) {
        cloud->clear();
        return false;
    }

    const std::size_t num_points = data_size / message.point_stride();
    cloud->clear();
    cloud->reserve(num_points);

    const double timestamp = stamp->sec() + stamp->nsec() * 1e-9;
    for (std::size_t i = 0; i < num_points; ++i) {
        const auto* data = data_buffer->Data() + i * stride;

        const float x = *reinterpret_cast<const float*>(data + x_offset);
        const float y = *reinterpret_cast<const float*>(data + y_offset);
        const float z = *reinterpret_cast<const float*>(data + z_offset);
        const float intensity = static_cast<float>(*reinterpret_cast<const uint8_t*>(data + reflectivity_offset));
        const uint8_t tag = *reinterpret_cast<const uint8_t*>(data + tag_offset);
        const uint8_t line = *reinterpret_cast<const uint8_t*>(data + line_offset);
        const double offset_time = static_cast<double>(*reinterpret_cast<const uint32_t*>(data + timestamp_offset)) / 1e9;

        cloud->push_back(slam_core::PointXYZIT(x, y, z, intensity, timestamp + offset_time));
    }

    return true;
}

inline bool DecodeCompressedImageMessage(
    const foxglove::CompressedImage& message,
    slam_core::Image& image_out)
{
    const auto* raw = message.data();
    const auto* stamp = message.timestamp();
    if (!raw || !stamp) {
        image_out = slam_core::Image();
        return false;
    }

    const double timestamp = stamp->sec() + stamp->nsec() * 1e-9;

    std::vector<std::uint8_t> buffer(raw->begin(), raw->end());
    const auto decoded_mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (decoded_mat.empty()) {
        image_out = slam_core::Image();
        return false;
    }

    image_out = slam_core::Image(decoded_mat, timestamp);
    return true;
}

inline bool ConvertImuMessage(const foxglove::Imu& message, slam_core::IMU& imu_out, const bool g_unit = true)
{
    const auto* angular_velocity = message.angular_velocity();
    const auto* linear_acceleration = message.linear_acceleration();
    const auto* stamp = message.timestamp();
    if (!angular_velocity || !linear_acceleration || !stamp) {
        imu_out = slam_core::IMU();
        return false;
    }

    const double timestamp = stamp->sec() + stamp->nsec() * 1e-9;

    Eigen::Vector3d angular_velocity_vec(angular_velocity->x(), angular_velocity->y(), angular_velocity->z());
    Eigen::Vector3d linear_acceleration_vec(0.0, 0.0, 0.0);
    if (g_unit) {
        linear_acceleration_vec = Eigen::Vector3d(linear_acceleration->x(), linear_acceleration->y(), linear_acceleration->z());
    } else {
        linear_acceleration_vec = Eigen::Vector3d(linear_acceleration->x(), linear_acceleration->y(), linear_acceleration->z());
    }

    imu_out = slam_core::IMU(angular_velocity_vec, linear_acceleration_vec, timestamp);
    return true;
}
}  // namespace ms_slam::slam_adapter
