#pragma once

#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "slam_core/local_mapping/local_mapper.hpp"
#include "slam_core/local_mapping/local_mapping_types.hpp"
#include "slam_core/map/voxel_map_voxelslam.hpp"

namespace ms_slam::slam_core::local_mapping
{
/**
 * @brief 基于 VoxelSLAM BALM 求解器的局部建图实现，独立维护 voxel 地图
 */
class BalmLocalMapper : public LocalMapper
{
  public:
    /**
     * @brief 使用配置构造局部建图实例
     * @param config 局部建图配置
     * @return 无
     */
    explicit BalmLocalMapper(const LocalMapperConfig& config);

    /**
     * @brief 析构时释放预积分与 voxel 资源
     * @param 无
     * @return 无
     */
    ~BalmLocalMapper() override;

    /**
     * @brief 推入一帧里程计输出
     * @param input 里程计输出数据
     * @return 无
     */
    void PushOdometryOutput(const OdometryOutput& input) override final;

    /**
     * @brief 尝试执行一次局部建图
     * @param 无
     * @return 成功返回结果，否则为空
     */
    std::optional<LocalMappingResult> TryProcess() override final;

    /**
     * @brief 清空内部状态与地图
     * @param 无
     * @return 无
     */
    void Reset() override final;

    void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) override final;

    void ExportStates(std::unordered_map<int, CommonState>& out) override final;

  private:
    /**
     * @brief 将里程计帧追加到滑窗并更新地图
     * @param input 里程计输出
     * @return 是否追加成功
     */
    bool AppendFrame(const OdometryOutput& input);

    void calcBodyVar(Eigen::Vector3d& pb, const float range_inc, const float degree_inc, Eigen::Matrix3d& var);
    void var_init(const PointCloudType::ConstPtr& pl_cur, voxelslam::PVecPtr& pptr);
    void pvec_update(voxelslam::PVecPtr pptr, const CommonState& x_curr, std::vector<Eigen::Vector3d>& pwld);

    void multi_recut(
        std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*>& feat_map,
        int win_count,
        std::vector<CommonState>& xs,
        voxelslam::LidarFactor& voxopt,
        std::vector<std::vector<voxelslam::SlideWindow*>>& sws);

    void multi_margi(
        std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*>& feat_map,
        double jour,
        int win_count,
        std::vector<CommonState>& xs,
        voxelslam::LidarFactor& voxopt,
        std::vector<voxelslam::SlideWindow*>& sw);

    /**
     * @brief 组装当前优化结果
     * @param optimized 是否进行了优化
     * @return 局部建图结果
     */
    std::optional<LocalMappingResult> BuildResult(bool optimized);

    LocalMapperConfig config_;
    std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*> voxel_nodes_;  ///< voxelmap 节点集合
    std::vector<std::vector<voxelslam::SlideWindow*>> window_pool_;                            ///< 滑窗缓冲池
    std::unordered_map<voxelslam::VOXEL_LOC, voxelslam::OctoTree*> slide_map_;
    std::deque<OdometryOutput> input_queue_;
    std::vector<CommonState> window_states_;
    std::vector<voxelslam::PVecPtr> window_points_;
    std::deque<voxelslam::IMU_PRE*> imu_factors_;
    std::vector<PointCloudType::Ptr> map_cloud_buffer_; ///< 可视化缓存
    std::unordered_map<int, CommonState> output_state_buffer_;  ///< 状态缓存
    voxelslam::LidarFactor lidar_factor_;
    voxelslam::LI_BA_Optimizer optimizer_;
    std::vector<int> mp_storage_;
    std::mutex mutex_;
    Eigen::Isometry3d T_i_l_;
    Eigen::MatrixXd hess_;
    Eigen::Vector3d last_pos_;
    CommonState x_curr_;
    CommonState opted_state_;
    int win_count_;
    int win_base_;
    int win_size_;
    double dept_err_;
    double beam_err_;
    double jour_;
    int next_keyframe_id_{0};
    bool initialized_{false};
    int thread_num_{5};
    const int mgsize_;
};

/**
 * @brief 创建 VoxelSLAM 风格的局部建图实例
 * @param config 局部建图配置
 * @return 局部建图指针
 */
std::unique_ptr<LocalMapper> CreateVoxelLocalMapper(const LocalMapperConfig& config);

}  // namespace ms_slam::slam_core::local_mapping
