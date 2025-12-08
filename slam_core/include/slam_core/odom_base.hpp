#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"
#include "slam_core/localmap_traits.hpp"
#include "slam_core/odom_common.hpp"
#include "slam_core/filter_state.hpp"
#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif

namespace ms_slam::slam_core
{
/**
 * @brief 非模板里程计抽象接口，Mapping 持有其指针以支持运行时切换实现
 */
class OdomBase
{
  public:
    /**
     * @brief 虚析构，保证多态释放
     */
    virtual ~OdomBase() = default;

    /**
     * @brief 处理一帧同步数据
     * @param sync_data 同步后的多模态数据
     * @return void
     */
    virtual void ProcessSyncData(const SyncData& sync_data) = 0;

    /**
     * @brief 导出可视化地图点云
     * @param out 点云缓存
     * @return void
     */
    virtual void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) = 0;

    /**
     * @brief 导出局部地图的点云视图
     * @param out 局部地图点云
     * @return void
     */
    virtual void ExportLocalMap(PointCloud<PointXYZDescriptor>::Ptr& out) = 0;

    /**
     * @brief 导出 LiDAR 状态序列，默认返回空
     * @param out 状态缓存
     * @return void
     */
    virtual void ExportLidarStates(std::vector<CommonState>& out) { out.clear(); }

#ifdef USE_PCL
    /**
     * @brief 导出 PCL 格式地图点云
     * @param out PCL 点云缓存
     * @return void
     */
    virtual void ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out) = 0;
#endif

    /**
     * @brief 是否完成初始化
     * @return 初始化状态
     */
    [[nodiscard]] virtual bool IsInitialized() const = 0;

    /**
     * @brief 当前帧索引
     * @return 帧编号
     */
    [[nodiscard]] virtual std::size_t FrameIndex() const = 0;

    /**
     * @brief 获取可视化与日志所需的状态视图
     * @return 状态摘要
     */
    [[nodiscard]] virtual CommonState GetState() const = 0;
};

/**
 * @brief 模板化基础实现，封装地图与通用资源管理
 * @tparam LocalMap 局部地图类型
 */
template<typename LocalMap>
class OdomBaseImpl : public OdomBase
{
  public:
    using LocalMapType = LocalMap;

    /**
     * @brief 构造基础里程计实现，初始化默认缓存
     * @return void
     */
    OdomBaseImpl();
    /**
     * @brief 默认析构函数
     * @return void
     */
    ~OdomBaseImpl() override = default;

    /**
     * @brief 从配置初始化公共参数与资源
     * @param cfg 全局配置
     * @return void
     */
    void InitializeFromConfig(const Config& cfg);

    /**
     * @brief 导出局部地图的点云视图（默认清空输出）
     * @param out 输出点云
     * @return void
     */
    void ExportLocalMap(PointCloud<PointXYZDescriptor>::Ptr& out) override;

    /**
     * @brief 导出局部地图原始实例（供派生类内部使用）
     * @param out 输出地图智能指针
     * @return void
     */
    void ExportLocalMap(std::unique_ptr<LocalMapType>& out);

    /**
     * @brief 导出可视化地图点云
     * @param out 点云缓冲
     * @return void
     */
    void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) override;

    /**
     * @brief 导出 LiDAR 状态序列（默认返回空）
     * @param out 状态缓存
     * @return void
     */
    void ExportLidarStates(std::vector<CommonState>& out) override { out.clear(); }

#ifdef USE_PCL
    /**
     * @brief 导出 PCL 地图点云
     * @param out PCL 点云缓存
     * @return void
     */
    void ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out) override;
#endif

    /**
     * @brief 是否完成初始化
     * @return 是否已完成初始化
     */
    [[nodiscard]] bool IsInitialized() const override;

    /**
     * @brief 当前帧索引
     * @return 当前帧编号
     */
    [[nodiscard]] std::size_t FrameIndex() const override;

  protected:
    /**
     * @brief 推入新地图点云（线程安全）
     * @param cloud 点云数据
     * @return void
     */
    void PushMapCloud(const PointCloudType::Ptr& cloud);

    /**
     * @brief 利用当前位姿与点云更新局部地图并维护可视化缓存
     * @param world_T_lidar 世界到 LiDAR 的变换
     * @param state_p 当前平移，用于地图更新
     * @param deskewed 去畸变点云
     * @param downsampled 下采样点云
     * @return void
     */
    void UpdateLocalMap(
        const Eigen::Isometry3d& world_T_lidar,
        const Eigen::Vector3d& state_p,
        const PointCloudType::Ptr& deskewed,
        const PointCloudType::Ptr& downsampled);

    /**
     * @brief 帧计数递增
     * @return void
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
OdomBaseImpl<LocalMap>::OdomBaseImpl()
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
void OdomBaseImpl<LocalMap>::InitializeFromConfig(const Config& cfg)
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
    spdlog::info("OdomBase initialized: lidar_cov {:.6f}", lidar_measurement_cov_);
}

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::ExportLocalMap(PointCloud<PointXYZDescriptor>::Ptr& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (out) {
        out->clear();
    }
}

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::ExportLocalMap(std::unique_ptr<LocalMapType>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.reset();
}

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::ExportMapCloud(std::vector<PointCloudType::Ptr>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.clear();
    if (!map_cloud_buffer_.empty()) {
        out.swap(map_cloud_buffer_);
    }
}

template<typename LocalMap>
bool OdomBaseImpl<LocalMap>::IsInitialized() const
{
    return initialized_.load(std::memory_order_relaxed);
}

template<typename LocalMap>
std::size_t OdomBaseImpl<LocalMap>::FrameIndex() const
{
    return frame_index_.load(std::memory_order_relaxed);
}

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::PushMapCloud(const PointCloudType::Ptr& cloud)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    map_cloud_buffer_.emplace_back(cloud);
}

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::UpdateLocalMap(
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
void OdomBaseImpl<LocalMap>::ExportPclMapCloud(std::vector<PointCloudT::Ptr>& out)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    out.clear();
    if (!pcl_map_cloud_buffer_.empty()) {
        out.swap(pcl_map_cloud_buffer_);
    }
}
#endif

template<typename LocalMap>
void OdomBaseImpl<LocalMap>::BumpFrame()
{
    frame_index_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace ms_slam::slam_core
