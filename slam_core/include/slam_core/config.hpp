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
    float blind;
    int point_filter_num;
};

struct MappingParams {
    bool gravity_align, check_satu;
    float satu_acc, satu_gyro, acc_norm;
    float down_size;
    float laser_point_cov, gyr_cov, acc_cov, b_acc_cov, b_gyr_cov;
    float plane_thr, fov_deg, DET_RANGE;
    Eigen::Vector3d gravity_init, gravity;
    Eigen::Vector3d extrinT;
    Eigen::Matrix3d extrinR;
    float keyframe_adding_dist_thres, keyframe_adding_ang_thres;
    float cube_len, vel_cov;
    bool save_in_advance;
};

struct LocalMapParams {
    float voxel_size, map_clipping_distance, plane_threshold;
    int max_points_per_voxel, voxel_neighborhood, knn_num, min_knn_num;
    int max_layer, max_iteration;
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
    LocalMapParams localmap_params;
    CameraParams camera_params;

    static Config& GetInstance()
    {
        static Config config;
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
