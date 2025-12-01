#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif
#include "slam_core/config.hpp"
#include "slam_core/estimator_base.hpp"
#include "slam_core/filter_estimator.hpp"
#include "slam_core/odom_common.hpp"
#include "slam_core/localmap_traits.hpp"

namespace ms_slam::slam_core
{
using DefaultLocalMap = VDBMap;
using DefaultEstimator = FilterEstimator<DefaultLocalMap>;

/**
 * @brief Odometry 模板类，依赖满足 EstimatorConcept 的估计器
 */
template<EstimatorConcept Estimator = DefaultEstimator>
class Odometry
{
  public:
    explicit Odometry();

    ~Odometry();

    Odometry(const Odometry&) = delete;
    Odometry& operator=(const Odometry&) = delete;

    void AddIMUData(const IMU& imu_data);

    void AddLidarData(const PointCloudType::ConstPtr& lidar_data);

    void AddImageData(const Image& image_data);

    [[nodiscard]] std::vector<SyncData> SyncPackages();

    void RunOdometry();

    void GetLidarState(typename Estimator::StatesType& buffer);

    void GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    void GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map);

    void Stop();

#ifdef USE_PCL
    void PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data);

    void GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer);
#endif

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    Estimator estimator_;  ///< 估计器封装

    std::unique_ptr<std::thread> odometry_thread_;  ///< 里程计线程

    double last_timestamp_imu_;

    std::atomic<bool> running_;  ///< 运行状态

#ifdef USE_PCL
    std::deque<PointCloudT::ConstPtr> pcl_lidar_buffer_;
#endif
};
}  // namespace ms_slam::slam_core

// 默认使用滤波估计器的类型别名
namespace ms_slam::slam_core
{
using FilterOdometry = Odometry<DefaultEstimator>;
}  // namespace ms_slam::slam_core
