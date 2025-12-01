#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"
#include "slam_core/estimator_base.hpp"
#include "slam_core/filter_state.hpp"

namespace ms_slam::slam_core
{
/**
 * @brief 基于滤波的前端估计器，提供 IMU 预测、去畸变与观测更新的封装
 */
template<typename LocalMap>
class FilterEstimator : public EstimatorBase<LocalMap>
{
  public:
    using StateType = FilterState;
    using StatesType = FilterStates;
    using LocalMapType = typename EstimatorBase<LocalMap>::LocalMapType;

    FilterEstimator();
    ~FilterEstimator() = default;

    /**
     * @brief 处理一帧同步数据，完成预测、去畸变、更新与地图维护
     * @param sync_data 同步数据
     */
    void ProcessSyncData(const SyncData& sync_data);

    /**
     * @brief 导出当前状态
     */
    [[nodiscard]] StateType GetStateSnapshot() const;

    /**
     * @brief 导出可视化地图点云
     * @param out 点云缓存
     */
    void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) { EstimatorBase<LocalMap>::ExportMapCloud(out); }

    /**
     * @brief 导出 LiDAR 状态序列
     * @param out 状态序列
     */
    void ExportLidarStates(StatesType& out);

    /**
     * @brief 导出局部地图占位接口
     * @param out 输出指针
     */
    void ExportLocalMap(std::unique_ptr<LocalMapType>& out) { EstimatorBase<LocalMap>::ExportLocalMap(out); }

#ifdef USE_PCL
    /**
     * @brief 导出 PCL 地图点云
     * @param out PCL 点云缓存
     */
    void ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out) { EstimatorBase<LocalMap>::ExportPclMapCloud(out); }
#endif

    /**
     * @brief 是否完成初始化
     */
    [[nodiscard]] bool IsInitialized() const { return EstimatorBase<LocalMap>::IsInitialized(); }

    /**
     * @brief 当前帧索引
     */
    [[nodiscard]] std::size_t FrameIndex() const { return EstimatorBase<LocalMap>::FrameIndex(); }

  private:
    /**
     * @brief 初始对准，统计静止 IMU 完成重力与偏置估计
     * @param sync_data 同步数据
     */
    void TryInitialize(const SyncData& sync_data);

    /**
     * @brief 处理 IMU 数据并维护状态缓存
     * @param sync_data 同步后的数据包
     */
    void ProcessImuData(const SyncData& sync_data);

    /**
     * @brief 去畸变点云
     * @param cloud 输入点云
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudType::Ptr Deskew(const PointCloudType::ConstPtr& cloud) const;

#ifdef USE_PCL
    /**
     * @brief PCL 版本去畸变
     * @param cloud 输入 PCL 点云
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudT::Ptr PCLDeskew(const PointCloudT::ConstPtr& cloud) const;
#endif

    /**
     * @brief 点面观测模型（滤波专用）
     */
    void ObsModel(StateType::ObsH& H, StateType::ObsZ& z, StateType::NoiseDiag& noise_inv);

    /**
     * @brief 执行观测更新
     */
    void UpdateWithModel();

    /**
     * @brief 更新局部地图与可视化缓存
     */
    void UpdateLocalMap();

    /**
     * @brief 推入新的 LiDAR 状态
     * @param state 状态数据
     */
    void PushLidarState(const StateType& state);

    StateType state_;                ///< 当前滤波状态
    StatesType imu_state_buffer_;    ///< IMU 时刻状态缓存
    StatesType lidar_state_buffer_;  ///< LiDAR 时刻状态缓存

    int init_imu_count_;             ///< 初始化累计 IMU 数
    Eigen::Vector3d init_gyro_avg_;  ///< 初始化陀螺均值
    Eigen::Vector3d init_accel_avg_; ///< 初始化加计均值
    double init_last_imu_stamp_;     ///< 初始化阶段最后时间戳

    const Config& cfg_;  ///< 配置引用，避免重复获取单例
};
}  // namespace ms_slam::slam_core
