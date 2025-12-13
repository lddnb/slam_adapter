#pragma once
#include <Eigen/Core>

namespace ms_slam::slam_core
{
class IMU
{
  public:
    IMU() = default;

    IMU(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, const double timestamp, const std::uint64_t index = 0)
    : linear_acceleration_(accel),
      angular_velocity_(gyro),
      timestamp_(timestamp),
      index_(index)
    {
    }
    [[nodiscard]] const Eigen::Vector3d& angular_velocity() const noexcept { return angular_velocity_; }
    [[nodiscard]] const Eigen::Vector3d& linear_acceleration() const noexcept { return linear_acceleration_; }
    [[nodiscard]] double timestamp() const noexcept { return timestamp_; }
    [[nodiscard]] std::uint64_t index() const noexcept { return index_; }

  private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d angular_velocity_;
    Eigen::Vector3d linear_acceleration_;
    double timestamp_;
    std::uint64_t index_;
};

inline IMU imu2baselink(const IMU& imu, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double dt) noexcept
{
    Eigen::Vector3d ang_vel_cg = R * imu.angular_velocity();
    static Eigen::Vector3d ang_vel_cg_prev = ang_vel_cg;

    Eigen::Vector3d lin_accel_cg = R * imu.linear_acceleration();
    lin_accel_cg = lin_accel_cg + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-t) + ang_vel_cg.cross(ang_vel_cg.cross(-t));

    ang_vel_cg_prev = ang_vel_cg;

    return IMU(ang_vel_cg, lin_accel_cg, imu.timestamp());
}

}  // namespace ms_slam::slam_core