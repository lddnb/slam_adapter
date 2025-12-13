#pragma once

#include <atomic>
#include <deque>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "slam_core/config.hpp"
#include "slam_core/map/map_traits.hpp"
#include "slam_core/odometry/filter_odom.hpp"
#include "slam_core/odometry/filter_state.hpp"
#include "slam_core/odometry/odom_base.hpp"
#include "slam_core/odometry/odom_common.hpp"
#include "slam_core/local_mapping/balm_local_mapper.hpp"

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

std::unique_ptr<local_mapping::LocalMapper> CreateLocalMapper();

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
    void GetOdomState(std::vector<CommonState>& buffer);

    /**
     * @brief 导出里程计地图点云
     * @param cloud_buffer 点云缓存
     * @return void
     */
    void GetOdomCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    /**
     * @brief 导出局部建图优化后的状态量
     * 
     * @param buffer 
     */
    void GetLocalState(std::vector<CommonState>& buffer);

    /**
     * @brief 导出局部建图优化后的地图点云
     * @param local_map 局部建图地图点云缓存
     * @return void
     */
    void GetLocalCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    /**
     * @brief 停止映射线程
     * @return void
     */
    void Stop();

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    std::unique_ptr<OdomBase> estimator_;  ///< 估计器封装
    std::unique_ptr<local_mapping::LocalMapper> local_mapper_;  ///< 局部建图器

    std::unique_ptr<std::thread> mapping_thread_;  ///< 建图/里程计线程

    double last_timestamp_imu_;
    std::uint64_t last_index_imu_;

    std::unordered_map<int, PointCloudType::ConstPtr> lidar_data_buffer_;  ///< 激光帧缓存
    std::vector<PointCloudType::Ptr> local_map_buffer_;                    ///< 局部建图地图缓存

    std::atomic<bool> running_;  ///< 运行状态
};
}  // namespace ms_slam::slam_core
