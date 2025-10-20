# pragma once
#include <Eigen/Core>

namespace ms_slam::slam_core
{
class IMU
{
public:
    IMU() = default;

    IMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp)
        : linear_acceleration_(accel), angular_velocity_(gyro), timestamp_(timestamp)
    {
    }
    [[nodiscard]] const Eigen::Vector3d& linear_acceleration() const noexcept { return linear_acceleration_; }
    [[nodiscard]] const Eigen::Vector3d& angular_velocity() const noexcept { return angular_velocity_; }
    [[nodiscard]] double timestamp() const noexcept { return timestamp_; }

private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d linear_acceleration_;
    Eigen::Vector3d angular_velocity_;
    double timestamp_;
};
} // namespace ms_slam::slam_core