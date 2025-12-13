#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include <spdlog/spdlog.h>

#include "slam_core/common_state.hpp"
#include "slam_core/odometry/odom_common.hpp"
#include "slam_core/sensor/imu.hpp"
#include "slam_core/sensor/point_cloud.hpp"

namespace ms_slam::slam_core::local_mapping
{
/**
 * @brief 局部建图配置，映射自 VoxelSLAM 关键超参
 */
struct LocalMapperConfig
{
    int window_size{10};                 ///< 滑窗大小，至少覆盖预积分与平面因子
    double voxel_size{1.0};             ///< voxel 尺寸
    double min_eigen_value{0.0025};     ///< 平面因子最小特征值阈值
    std::vector<double> plane_eigen_value_thresholds{4.0, 4.0, 4.0, 4.0};  ///< 分层平面判定比值
    int max_layer{2};                   ///< voxel 分裂深度
    int max_points{100};                ///< 单 voxel 最大点数
    int min_ba_point{1};               ///< 触发平面因子所需的最少点数
};

/**
 * @brief 里程计输出给局部建图的输入数据
 */
struct OdometryOutput
{
    int index;                                      ///< 里程计帧索引
    CommonState state;                              ///< 里程计估计的 CommonState
    PointCloudType::ConstPtr orig_cloud;            ///< 原始去畸变点云
    PointCloudType::ConstPtr cloud;                 ///< 去畸变降采样后的点云
    std::deque<IMU> imu_buffer;                     ///< 覆盖该帧时间范围的 IMU 数据
};

/**
 * @brief 局部建图输出的优化结果
 */
struct LocalMappingResult
{
    int index;
    CommonState optimized_state;                ///< 与里程计时戳对齐的优化结果
    std::vector<CommonState> window_states;     ///< 滑窗内的状态序列
    bool is_keyframe{false};                    ///< 是否生成了关键帧
};

}  // namespace ms_slam::slam_core::local_mapping
