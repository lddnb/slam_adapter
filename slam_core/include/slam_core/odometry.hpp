#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

#ifdef USE_PCL
#include "slam_core/PCL.hpp"
#endif
#include "slam_core/config.hpp"
#include "slam_core/filter_estimator.hpp"
#include "slam_core/odom_common.hpp"
#include "slam_core/localmap_traits.hpp"

namespace ms_slam::slam_core
{
using DefaultLocalMap = VDBMap;
using DefaultEstimator = FilterEstimator<DefaultLocalMap>;

template<typename Estimator = DefaultEstimator, typename LocalMap = DefaultLocalMap>
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

    /**
     * @brief 使用通用去畸变接口，返回对齐参考时刻的点云
     * @param cloud 输入点云
     * @param state 当前滤波状态（参考时刻）
     * @param buffer 状态时间序列，用于插值
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudType::Ptr Deskew(const PointCloudType::ConstPtr& cloud, const typename Estimator::StateType& state, const typename Estimator::StatesType& buffer) const;

    void ProcessImuData(const SyncData& sync_data);

    void Initialize(const SyncData& sync_data);

    void RunOdometry();

    /**
     * @brief 构建点云观测模型并输出噪声逆
     * @param H 观测雅可比
     * @param z 观测残差
     * @param noise_inv 噪声协方差对角的逆（可为统一值或逐点值）
     */
    void ObsModel(typename Estimator::StateType::ObsH& H, typename Estimator::StateType::ObsZ& z, typename Estimator::StateType::NoiseDiag& noise_inv);

#if defined(USE_VOXELMAP)
    void VoxelMapObsModel(typename Estimator::StateType::ObsH& H, typename Estimator::StateType::ObsZ& z, typename Estimator::StateType::NoiseDiag& noise_inv);
#endif

    void GetLidarState(typename Estimator::StatesType& buffer);

    void GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer);

    void GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map);

    void Stop();

#ifdef USE_PCL
    void PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data);

    /**
     * @brief PCL 点云去畸变
     * @param cloud 输入点云
     * @param state 当前滤波状态（参考时刻）
     * @param buffer 状态缓存
     * @return 去畸变后的点云
     */
    [[nodiscard]] PointCloudT::Ptr PCLDeskew(const PointCloudT::ConstPtr& cloud, const typename Estimator::StateType& state, const typename Estimator::StatesType& buffer) const;

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

    std::unique_ptr<LocalMap> local_map_;
#if defined(USE_VOXELMAP)
    std::vector<Eigen::Matrix3d> var_down_body_;
#endif

    double last_timestamp_imu_;

    std::atomic<bool> running_;  ///< 运行状态

    typename Estimator::StateType state_;                ///< 当前里程计状态
    typename Estimator::StatesType imu_state_buffer_;    ///< imu时刻状态缓存
    typename Estimator::StatesType lidar_state_buffer_;  ///< lidar时刻状态缓存
    bool initialized_;                 ///< 是否初始化

    double imu_scale_factor_;  ///< 平均加速度
    std::mutex state_mutex_;   ///< 状态互斥锁

    std::vector<PointCloudType::Ptr> map_cloud_buffer_;  ///< 同步数据列表

    PointCloudType::Ptr deskewed_cloud_;
    PointCloudType::Ptr downsampled_cloud_;

    Eigen::Isometry3d T_i_l;

    std::size_t frame_index_;  ///< 帧索引

    double lidar_measurement_cov_;

    LocalMapParams localmap_params_;  ///< knn搜索数

#ifdef USE_PCL
    std::deque<PointCloudT::ConstPtr> pcl_lidar_buffer_;
    PointCloudT::Ptr pcl_deskewed_cloud_;
    PointCloudT::Ptr pcl_downsampled_cloud_;
    std::vector<PointCloudT::Ptr> pcl_map_cloud_buffer_;
#endif
};
}  // namespace ms_slam::slam_core

// 默认使用滤波估计器的类型别名
namespace ms_slam::slam_core
{
using FilterOdometry = Odometry<DefaultEstimator, DefaultLocalMap>;
}  // namespace ms_slam::slam_core
