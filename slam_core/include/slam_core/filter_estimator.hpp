#pragma once

#include <optional>
#include <string_view>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"
#include "slam_core/filter_state.hpp"
#include "slam_core/odom_common.hpp"

namespace ms_slam::slam_core
{
/**
 * @brief 基于滤波的前端估计器，提供 IMU 预测、去畸变与观测更新的封装
 */
template<typename LocalMap>
class FilterEstimator
{
  public:
    using StateType = FilterState;
    using StatesType = FilterStates;

    FilterEstimator() = default;
    ~FilterEstimator() = default;

    /**
     * @brief 处理 IMU 数据并维护状态缓存
     * @param sync_data 同步后的数据包
     * @param state 当前滤波状态
     * @param imu_buffer IMU 状态缓存
     * @param lidar_buffer LiDAR 时刻状态缓存
     * @param imu_scale_factor IMU 加速度比例因子
     * @param state_mutex 状态缓存互斥锁（用于 lidar_buffer 写入）
     */
    void ProcessImuData(
        const SyncData& sync_data,
        StateType& state,
        StatesType& imu_buffer,
        StatesType& lidar_buffer,
        double imu_scale_factor,
        std::mutex& state_mutex);

    /**
     * @brief 去畸变点云
     * @param cloud 输入点云
     * @param state 参考状态
     * @param buffer IMU 状态缓存
     * @param T_i_l IMU 到 LiDAR 外参
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudType::Ptr Deskew(
        const PointCloudType::ConstPtr& cloud,
        const StateType& state,
        const StatesType& buffer,
        const Eigen::Isometry3d& T_i_l) const;

#ifdef USE_PCL
    /**
     * @brief PCL 版本去畸变
     * @param cloud 输入 PCL 点云
     * @param state 参考状态
     * @param buffer IMU 状态缓存
     * @param T_i_l 外参
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudT::Ptr PCLDeskew(
        const PointCloudT::ConstPtr& cloud,
        const StateType& state,
        const StatesType& buffer,
        const Eigen::Isometry3d& T_i_l) const;
#endif

    /**
     * @brief 点面观测模型（滤波专用）
     */
    void ObsModel(
        const PointCloudType::Ptr& downsampled_cloud,
#ifdef USE_PCL
        const PointCloudT::Ptr& pcl_downsampled_cloud,
#endif
        LocalMap& local_map,
        const Eigen::Isometry3d& T_i_l,
        const LocalMapParams& localmap_params,
        std::size_t frame_index,
        double lidar_measurement_cov,
        const StateType& state,
        StateType::ObsH& H,
        StateType::ObsZ& z,
        StateType::NoiseDiag& noise_inv) const;

    /**
     * @brief 按名称触发观测更新
     * @param state 状态
     * @param name 观测模型名称
     */
    void UpdateWithModel(StateType& state, std::string_view name);
};
}  // namespace ms_slam::slam_core
