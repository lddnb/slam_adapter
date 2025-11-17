#pragma once

#define USE_VOXELMAP

#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif
#include "slam_core/imu.hpp"
#include "slam_core/point_cloud.hpp"
#include "slam_core/image.hpp"
#include "slam_core/state.hpp"
#ifdef USE_IKDTREE
#include "slam_core/ikd-Tree/ikd_Tree_impl.h"
#elif defined(USE_VDB)
#include "slam_core/VDB_map.hpp"
#elif defined(USE_HASHMAP)
#include "slam_core/hash_map.hpp"
#elif defined(USE_VOXELMAP)
#include "slam_core/voxel_map.hpp"
#endif

namespace ms_slam::slam_core
{
using PointType = PointXYZITDescriptor;
using PointCloudType = PointCloud<PointType>;

struct SyncData {
    PointCloudType::ConstPtr lidar_data;
#ifdef USE_PCL
    PointCloudT::ConstPtr pcl_lidar_data;
#endif
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

    /**
     * @brief 构建点云观测模型并输出噪声逆
     * @param H 观测雅可比
     * @param z 观测残差
     * @param noise_inv 噪声协方差对角的逆（可为统一值或逐点值）
     */
    void ObsModel(State::ObsH& H, State::ObsZ& z, State::NoiseDiag& noise_inv);

#if defined(USE_VOXELMAP)
    void VoxelMapObsModel(State::ObsH& H, State::ObsZ& z, State::NoiseDiag& noise_inv);
#endif

    void GetLidarState(States& buffer);

    void GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    void GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map);

    void Stop();

#ifdef USE_PCL
    void PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data);

    [[nodiscard]] PointCloudT::Ptr PCLDeskew(const PointCloudT::ConstPtr& cloud, const State& state, const States& buffer) const;

    void GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer);
#endif

  private:
    std::deque<IMU> imu_buffer_;                         ///< imu缓存
    std::deque<PointCloudType::ConstPtr> lidar_buffer_;  ///< lidar缓存
    std::deque<Image> image_buffer_;                     ///< 图像缓存

    std::mutex data_mutex_;  ///< 数据互斥锁

    bool visual_enable_;  ///< 可视化开关

    std::unique_ptr<std::thread> odometry_thread_;  ///< 里程计线程

#ifdef USE_IKDTREE
    std::unique_ptr<ikdtreeNS::KD_TREE<ikdtreeNS::ikdTree_PointType> > local_map_;  ///< 局部地图
#elif defined(USE_VDB)
    std::unique_ptr<VDBMap> local_map_;
#elif defined(USE_HASHMAP)
    std::unique_ptr<voxelHashMap> local_map_;  ///< 局部地图
#elif defined(USE_VOXELMAP)
    std::unique_ptr<VoxelMap> local_map_;
    std::vector<Eigen::Matrix3d> var_down_body_;
#endif

    double last_timestamp_imu_;

    std::atomic<bool> running_;  ///< 运行状态

    State state_;                ///< 当前里程计状态
    States imu_state_buffer_;    ///< imu时刻状态缓存
    States lidar_state_buffer_;  ///< lidar时刻状态缓存
    bool initialized_;           ///< 是否初始化

    double imu_scale_factor_;  ///< 平均加速度
    std::mutex state_mutex_;   ///< 状态互斥锁

    std::vector<PointCloudType::Ptr> map_cloud_buffer_;  ///< 同步数据列表

    PointCloudType::Ptr deskewed_cloud_;
    PointCloudType::Ptr downsampled_cloud_;

    Eigen::Isometry3d T_i_l;

    std::size_t frame_index_;  ///< 帧索引

    double lidar_measurement_cov_;

#ifdef USE_PCL
    std::deque<PointCloudT::ConstPtr> pcl_lidar_buffer_;
    PointCloudT::Ptr pcl_deskewed_cloud_;
    PointCloudT::Ptr pcl_downsampled_cloud_;
    std::vector<PointCloudT::Ptr> pcl_map_cloud_buffer_;
#endif
};
}  // namespace ms_slam::slam_core
