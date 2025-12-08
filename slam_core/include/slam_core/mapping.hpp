#pragma once

#include <atomic>
#include <deque>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif
#include "slam_core/config.hpp"
#include "slam_core/filter_state.hpp"
#include "slam_core/localmap_traits.hpp"
#include "slam_core/odom_base.hpp"
#include "slam_core/odom_common.hpp"
#include "slam_core/filter_odom.hpp"

namespace ms_slam::slam_core
{
/**
 * @brief 运行时可切换的里程计后端类型
 */
enum class OdomType
{
    kFilterVdb = 0,
    kFilterVoxelHash = 1,
    kFilterOctree = 2
};

/**
 * @brief 创建指定后端的里程计估计器
 * @param type 后端类型
 * @return 里程计智能指针
 */
std::unique_ptr<OdomBase> CreateOdomEstimator(OdomType type);

/**
 * @brief Mapping 运行时多态类，内部持有 OdomBase 指针以动态切换里程计
 */
class Mapping
{
  public:
    /**
     * @brief 构造 Mapping 管线，初始化内部状态
     * @param type 里程计选择
     * @return void
     */
    explicit Mapping(OdomType type = OdomType::kFilterVdb);

    /**
     * @brief 析构函数，释放线程与资源
     * @return void
     */
    ~Mapping();

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    /**
     * @brief 推入 IMU 数据
     * @param imu_data IMU 数据
     * @return void
     */
    void AddIMUData(const IMU& imu_data);

    /**
     * @brief 推入 LiDAR 数据
     * @param lidar_data 去畸变前的点云指针
     * @return void
     */
    void AddLidarData(const PointCloudType::ConstPtr& lidar_data);

    /**
     * @brief 推入图像数据
     * @param image_data 图像帧
     * @return void
     */
    void AddImageData(const Image& image_data);

    /**
     * @brief 同步多源数据以供估计器处理
     * @return 同步后的数据包序列
     */
    [[nodiscard]] std::vector<SyncData> SyncPackages();

    /**
     * @brief 运行映射线程主循环
     * @return void
     */
    void RunMapping();

    /**
     * @brief 导出 LiDAR 时刻位姿序列
     * @param buffer 外部缓存
     * @return void
     */
    void GetLidarState(std::vector<CommonState>& buffer);

    /**
     * @brief 导出地图点云
     * @param cloud_buffer 点云缓存
     * @return void
     */
    void GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    /**
     * @brief 导出局部地图
     * @param local_map 局部地图输出
     * @return void
     */
    void GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map);

    /**
     * @brief 停止映射线程
     * @return void
     */
    void Stop();

#ifdef USE_PCL
    /**
     * @brief 推入 PCL 格式点云数据
     * @param lidar_data PCL 点云
     * @return void
     */
    void PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data);

    /**
     * @brief 导出 PCL 格式地图
     * @param cloud_buffer 点云缓存
     * @return void
     */
    void GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer);
#endif

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    std::unique_ptr<OdomBase> estimator_;  ///< 估计器封装

    std::unique_ptr<std::thread> mapping_thread_;  ///< 建图/里程计线程

    double last_timestamp_imu_;
    std::uint64_t last_index_imu_;

    std::atomic<bool> running_;  ///< 运行状态

#ifdef USE_PCL
    std::deque<PointCloudT::ConstPtr> pcl_lidar_buffer_;
#endif
};
}  // namespace ms_slam::slam_core

