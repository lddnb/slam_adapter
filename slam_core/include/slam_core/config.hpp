#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace ms_slam::slam_core
{

struct CommonParams {
    std::string lid_topic, imu_topic, visloop_topic, rtk_topic;
    std::vector<std::string> img_topics;
    bool space_down_sample;       ///< 稀疏点云的降采样参数
    float time_lag_imu_to_lidar;  ///< imu和雷达的时间戳差值
    float time_lag_img_to_lidar;  ///< 图像和雷达的时间戳差值
    bool pcd_save_en;             ///< 点云保存flag
    bool path_en, scan_pub_en, scan_body_pub_en, opt_map_pub_en;
    int lidar_type;
    bool render_en;  ///< 点云着色flag
    bool dense_pc;   ///< 发布稠密点云flag
};

struct MappingParams {
    bool non_station_start, extrinsic_est_en, gravity_align, prop_at_freq_of_imu, check_satu;
    int init_map_size;
    float imu_time_inte, satu_acc, satu_gyro, acc_norm;
    float laser_point_cov, acc_cov_output, gyr_cov_output, b_acc_cov, b_gyr_cov;
    float imu_meas_acc_cov, imu_meas_omg_cov, gyr_cov, acc_cov;
    float plane_thr, match_s, fov_deg, DET_RANGE;
    Eigen::Vector3d gravity_init, gravity;
    Eigen::Vector3d extrinT;
    Eigen::Matrix3d extrinR;
    float keyframe_adding_dist_thres, keyframe_adding_ang_thres;
    float filter_size_surf_min, filter_size_map_min;
    float cube_len, vel_cov;
    bool save_in_advance;
};

struct VoxelParams {
    float voxel_size, plannar_threshold;
    std::vector<int> layer_point_size;
    float ranging_cov, angle_cov, acc_cov, gyr_cov, b_acc_cov, b_gyr_cov;
    int max_layer, max_points_size, max_cov_points_size, max_iteration;
};

struct CameraParams {
    std::vector<Eigen::Matrix4d> extrinsic_mats;            // 外参矩阵
    std::vector<Eigen::Matrix3d> rotate_mats, camera_mats;  // 旋转矩阵，内参矩阵
    std::vector<Eigen::VectorXd> transform_vecs;            // 平移向量
    std::vector<Eigen::VectorXd> dist_coeffs;               // 畸变矩阵
    std::vector<Eigen::Vector2d> img_sizes;
    bool fisheye_en;
    std::string image_save_path;
    int camera_num;
};

class Config
{
  public:
    CommonParams common_params;
    MappingParams mapping_params;
    VoxelParams voxel_params;
    CameraParams camera_params;

    static Config* GetInstance()
    {
        static Config* config = new Config();
        return config;
    }

  private:
    // Singleton pattern
    Config() = default;

    // Delete copy/move so extra instances can't be created/moved.
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;
};

}  // namespace ms_slam::slam_core
