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
#include "slam_core/odom_base.hpp"
#include "slam_core/filter_state.hpp"

namespace ms_slam::slam_core
{
/**
 * @brief 基于滤波的里程计估计器，提供 IMU 预测、去畸变与观测更新的封装
 */
template<typename LocalMap>
class FilterOdom : public OdomBaseImpl<LocalMap>
{
  public:
    using StateType = FilterState;
    using StatesType = FilterStates;
    using LocalMapType = typename OdomBaseImpl<LocalMap>::LocalMapType;

    FilterOdom();
    ~FilterOdom() = default;

    /**
     * @brief 处理一帧同步数据，完成预测、去畸变、更新与地图维护
     * @param sync_data 同步数据
     * @return void
     */
    void ProcessSyncData(const SyncData& sync_data) override;

    /**
     * @brief 导出当前状态
     * @return 当前滤波状态
     */
    [[nodiscard]] StateType GetStateSnapshot() const;

    /**
     * @brief 获取状态
     * @return 状态摘要
     */
    [[nodiscard]] CommonState GetState() const override;

    /**
     * @brief 导出可视化地图点云
     * @param out 点云缓存
     * @return void
     */
    void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) override { OdomBaseImpl<LocalMap>::ExportMapCloud(out); }

    /**
     * @brief 导出 LiDAR 状态序列
     * @param out 状态序列
     * @return void
     */
    void ExportLidarStates(std::vector<CommonState>& out) override;

    /**
     * @brief 导出局部地图占位接口
     * @param out 输出点云
     * @return void
     */
    void ExportLocalMap(PointCloud<PointXYZDescriptor>::Ptr& out) override { OdomBaseImpl<LocalMap>::ExportLocalMap(out); }

    /**
     * @brief 导出局部地图实例，便于派生逻辑复用
     * @param out 输出地图智能指针
     * @return void
     */
    void ExportLocalMap(std::unique_ptr<LocalMapType>& out) { OdomBaseImpl<LocalMap>::ExportLocalMap(out); }

#ifdef USE_PCL
    /**
     * @brief 导出 PCL 地图点云
     * @param out PCL 点云缓存
     * @return void
     */
    void ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out) override { OdomBaseImpl<LocalMap>::ExportPclMapCloud(out); }
#endif

    /**
     * @brief 是否完成初始化
     * @return 是否已完成初始化
     */
    [[nodiscard]] bool IsInitialized() const override { return OdomBaseImpl<LocalMap>::IsInitialized(); }

    /**
     * @brief 当前帧索引
     * @return 当前帧编号
     */
    [[nodiscard]] std::size_t FrameIndex() const override { return OdomBaseImpl<LocalMap>::FrameIndex(); }

  private:
    /**
     * @brief 初始对准，统计静止 IMU 完成重力与偏置估计
     * @param sync_data 同步数据
     * @return void
     */
    void TryInitialize(const SyncData& sync_data);

    /**
     * @brief 处理 IMU 数据并维护状态缓存
     * @param sync_data 同步后的数据包
     * @return void
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
     * @param H 观测雅可比矩阵
     * @param z 观测残差向量
     * @param noise_inv 噪声对角阵的逆
     * @return void
     */
    void ObsModel(StateType::ObsH& H, StateType::ObsZ& z, StateType::NoiseDiag& noise_inv);

    /**
     * @brief 执行观测更新
     * @return void
     */
    void UpdateWithModel();

    /**
     * @brief 更新局部地图与可视化缓存
     * @return void
     */
    void UpdateLocalMap();

    /**
     * @brief 推入新的 LiDAR 状态
     * @param state 状态数据
     * @return void
     */
    void PushLidarState(const StateType& state);

    StateType state_;                ///< 当前滤波状态
    StatesType imu_state_buffer_;    ///< IMU 时刻状态缓存
    std::vector<CommonState> lidar_state_buffer_;  ///< LiDAR 时刻状态缓存

    int init_imu_count_;             ///< 初始化累计 IMU 数
    Eigen::Vector3d init_gyro_avg_;  ///< 初始化陀螺均值
    Eigen::Vector3d init_accel_avg_; ///< 初始化加计均值
    double init_last_imu_stamp_;     ///< 初始化阶段最后时间戳

    const Config& cfg_;  ///< 配置引用，避免重复获取单例
};
}  // namespace ms_slam::slam_core
