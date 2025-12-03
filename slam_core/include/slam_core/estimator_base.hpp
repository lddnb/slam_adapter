#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <memory>
#include <mutex>
#include <utility>
#include <type_traits>
#include <vector>

#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"
#include "slam_core/localmap_traits.hpp"
#include "slam_core/odom_common.hpp"
#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif

namespace ms_slam::slam_core
{
/**
 * @brief Estimator 接口约束，编译期确保估计器具备必要方法
 */
template<typename T>
concept EstimatorConcept = requires(
    T estimator,
    const SyncData& sync_data,
    std::vector<PointCloudType::Ptr>& map_clouds,
    typename T::StatesType& lidar_states,
    std::unique_ptr<typename T::LocalMapType>& local_map) {
    typename T::StateType;
    typename T::StatesType;
    typename T::LocalMapType;
    { estimator.ProcessSyncData(sync_data) };
    { estimator.GetStateSnapshot() } -> std::same_as<typename T::StateType>;
    { estimator.ExportMapCloud(map_clouds) };
    { estimator.ExportLidarStates(lidar_states) };
    { estimator.ExportLocalMap(local_map) };
#ifdef USE_PCL
    { estimator.ExportPclMapCloud(std::declval<std::vector<PointCloudT::Ptr>&>()) };
#endif
};

/**
 * @brief 估计器基础类，集中管理状态、局部地图与公共配置
 * @tparam LocalMap 局部地图类型
 */
template<typename LocalMap>
class EstimatorBase
{
  public:
    using LocalMapType = LocalMap;

    EstimatorBase();
    ~EstimatorBase() = default;

    /**
     * @brief 从配置初始化公共参数与资源
     * @param cfg 全局配置
     */
    void InitializeFromConfig(const Config& cfg);

    /**
     * @brief 导出局部地图（默认返回空指针，占位接口）
     * @param out 输出地图指针
     */
    void ExportLocalMap(std::unique_ptr<LocalMapType>& out);

    /**
     * @brief 导出可视化地图点云
     * @param out 点云缓冲
     */
    void ExportMapCloud(std::vector<PointCloudType::Ptr>& out);

#ifdef USE_PCL
    /**
     * @brief 导出 PCL 地图点云
     * @param out PCL 点云缓存
     */
    void ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out);
#endif

    /**
     * @brief 是否完成初始化
     */
    [[nodiscard]] bool IsInitialized() const;

    /**
     * @brief 当前帧索引
     */
    [[nodiscard]] std::size_t FrameIndex() const;

  protected:
    /**
     * @brief 推入新地图点云（线程安全）
     * @param cloud 点云数据
     */
    void PushMapCloud(const PointCloudType::Ptr& cloud);

    /**
     * @brief 利用当前位姿与点云更新局部地图并维护可视化缓存
     * @param world_T_lidar 世界到 LiDAR 的变换
     * @param state_p 当前平移，用于地图更新
     * @param deskewed 去畸变点云
     * @param downsampled 下采样点云
     */
    void UpdateLocalMap(
        const Eigen::Isometry3d& world_T_lidar,
        const Eigen::Vector3d& state_p,
        const PointCloudType::Ptr& deskewed,
        const PointCloudType::Ptr& downsampled);

    /**
     * @brief 帧计数递增
     */
    void BumpFrame();

    std::unique_ptr<LocalMapType> local_map_;  ///< 局部地图
    LocalMapParams localmap_params_;           ///< 地图参数

    Eigen::Isometry3d T_i_l_;  ///< IMU 到 LiDAR 外参

    double lidar_measurement_cov_;  ///< 激光测量噪声
    double imu_scale_factor_;       ///< 加速度标定系数

    std::atomic<std::size_t> frame_index_;  ///< 帧计数
    std::atomic<bool> initialized_;         ///< 初始化标记

    PointCloudType::Ptr deskewed_cloud_;    ///< 去畸变点云
    PointCloudType::Ptr downsampled_cloud_; ///< 下采样点云

#ifdef USE_PCL
    PointCloudT::Ptr pcl_deskewed_cloud_;                ///< PCL 去畸变点云
    PointCloudT::Ptr pcl_downsampled_cloud_;             ///< PCL 下采样点云
    std::vector<PointCloudT::Ptr> pcl_map_cloud_buffer_; ///< PCL 可视化缓存
#endif

    std::vector<PointCloudType::Ptr> map_cloud_buffer_; ///< 可视化缓存

    mutable std::mutex state_mutex_; ///< 状态与地图互斥锁
};

template<typename LocalMap>
EstimatorBase<LocalMap>::EstimatorBase()
    : T_i_l_(Eigen::Isometry3d::Identity()),
      lidar_measurement_cov_(1.0),
      imu_scale_factor_(1.0),
      frame_index_(0),
      initialized_(false)
{
    deskewed_cloud_ = std::make_shared<PointCloudType>();
    downsampled_cloud_ = std::make_shared<PointCloudType>();
#ifdef USE_PCL
    pcl_deskewed_cloud_ = PointCloudT::Ptr(new PointCloudT);
    pcl_downsampled_cloud_ = PointCloudT::Ptr(new PointCloudT);
#endif
}

template<typename LocalMap>
void EstimatorBase<LocalMap>::InitializeFromConfig(const Config& cfg)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    localmap_params_ = cfg.localmap_params;
    lidar_measurement_cov_ = std::max(static_cast<double>(cfg.mapping_params.laser_point_cov), 1e-6);

    T_i_l_.setIdentity();
    T_i_l_.linear() = cfg.mapping_params.extrinR;
    T_i_l_.translation() = cfg.mapping_params.extrinT;

    local_map_ = MapTraits<LocalMapType>::Create(localmap_params_);
    imu_scale_factor_ = 1.0;
    frame_index_.store(0);
    initialized_.store(false);
    map_cloud_buffer_.clear();
#ifdef USE_PCL
    pcl_map_cloud_buffer_.clear();
#endif
    spdlog::info("EstimatorBase initialized: lidar_cov {:.6f}", lidar_measurement_cov_);
}

template<typename LocalMap>
void EstimatorBase<LocalMap>::ExportLocalMap(std::unique_ptr<LocalMapType>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.reset();
}

template<typename LocalMap>
void EstimatorBase<LocalMap>::ExportMapCloud(std::vector<PointCloudType::Ptr>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.clear();
    if (!map_cloud_buffer_.empty()) {
        out.swap(map_cloud_buffer_);
    }
}

template<typename LocalMap>
bool EstimatorBase<LocalMap>::IsInitialized() const
{
    return initialized_.load(std::memory_order_relaxed);
}

template<typename LocalMap>
std::size_t EstimatorBase<LocalMap>::FrameIndex() const
{
    return frame_index_.load(std::memory_order_relaxed);
}

template<typename LocalMap>
void EstimatorBase<LocalMap>::PushMapCloud(const PointCloudType::Ptr& cloud)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    map_cloud_buffer_.emplace_back(cloud);
}

template<typename LocalMap>
void EstimatorBase<LocalMap>::UpdateLocalMap(
    const Eigen::Isometry3d& world_T_lidar,
    const Eigen::Vector3d& state_p,
    const PointCloudType::Ptr& deskewed,
    const PointCloudType::Ptr& downsampled)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (deskewed) {
        auto cloud_world = deskewed->clone();
        cloud_world->transform(world_T_lidar);
        map_cloud_buffer_.emplace_back(cloud_world);
    }

#ifdef USE_PCL
    if (pcl_deskewed_cloud_) {
        PointCloudT::Ptr pcl_clone(new PointCloudT(*pcl_deskewed_cloud_));
        const Eigen::Matrix4f tf = world_T_lidar.matrix().cast<float>();
        pcl::transformPointCloud(*pcl_clone, *pcl_clone, tf);
        pcl_map_cloud_buffer_.emplace_back(pcl_clone);
    }
#endif

    if (downsampled && local_map_) {
        auto ori_points = downsampled->positions_vec3();
        if constexpr (std::is_same_v<LocalMap, thuni::Octree>) {
            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> local_points;
            local_points.assign(ori_points.begin(), ori_points.end());
            MapTraits<LocalMap>::Update(*local_map_, local_points, world_T_lidar, state_p);
            spdlog::info("local map add {} points", local_points.size());
        } else {
            std::vector<Eigen::Vector3f> local_points;
            local_points.assign(ori_points.begin(), ori_points.end());
            MapTraits<LocalMap>::Update(*local_map_, local_points, world_T_lidar, state_p);
            spdlog::info("local map add {} points", local_points.size());
        }
        frame_index_.fetch_add(1, std::memory_order_relaxed);
    }
}

#ifdef USE_PCL
template<typename LocalMap>
void EstimatorBase<LocalMap>::ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.clear();
    if (!pcl_map_cloud_buffer_.empty()) {
        out.swap(pcl_map_cloud_buffer_);
    }
}
#endif

template<typename LocalMap>
void EstimatorBase<LocalMap>::BumpFrame()
{
    frame_index_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace ms_slam::slam_core
