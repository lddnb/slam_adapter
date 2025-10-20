#pragma once
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

#include "slam_core/imu.hpp"
#include "slam_core/point_cloud.hpp"
#include "slam_core/image.hpp"

namespace ms_slam::slam_core
{
using PointType = PointXYZITDescriptor;
using PointCloudType = PointCloud<PointType>;

struct SyncData {
    PointCloudType::ConstPtr lidar_data;
    double lidar_beg_time;
    double lidar_end_time;
    Image image_data;
    std::vector<IMU> imu_data;
};

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

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    std::unique_ptr<std::thread> odometry_thread_;  ///< 里程计线程

    double last_timestamp_imu_;

    std::atomic<bool> running_;  ///< 运行状态
};
}  // namespace ms_slam::slam_core