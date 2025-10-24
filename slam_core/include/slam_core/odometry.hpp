#pragma once
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

#include "slam_core/imu.hpp"
#include "slam_core/point_cloud.hpp"
#include "slam_core/image.hpp"
#include "slam_core/state.hpp"

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

    [[nodiscard]] PointCloudType::Ptr Deskew(const PointCloudType::ConstPtr& cloud, const State& state, const States& buffer) const;

    void ProcessImuData(const SyncData& sync_data);

    void Initialize(const SyncData& sync_data);

    void RunOdometry();

    void GetLidarState(States& buffer);

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    std::unique_ptr<std::thread> odometry_thread_;  ///< 里程计线程

    double last_timestamp_imu_;

    std::atomic<bool> running_;  ///< 运行状态

    State state_;          ///< 当前里程计状态
    States imu_state_buffer_;  ///< imu时刻状态缓存
    States lidar_state_buffer_;  ///< lidar时刻状态缓存
    bool initialized_;     ///< 是否初始化

    Eigen::Vector3d mean_acc_;  ///< 平均加速度
    std::mutex state_mutex_;   ///< 状态互斥锁
};
}  // namespace ms_slam::slam_core