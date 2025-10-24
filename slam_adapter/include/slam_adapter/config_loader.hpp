#pragma once

#include <algorithm>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <slam_core/config.hpp>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace ms_slam::slam_adapter
{

namespace detail
{

template <typename T>
inline void AssignIfPresent(const YAML::Node& node, const std::string& key, T& target)
{
    if (const auto value = node[key]) {
        if constexpr (std::is_floating_point_v<T>) {
            target = static_cast<T>(value.as<double>());
        } else if constexpr (std::is_same_v<T, std::string>) {
            target = value.as<std::string>();
        } else {
            target = value.as<T>();
        }
    }
}

template <typename T>
inline std::optional<std::vector<T>> SequenceToVector(const YAML::Node& node)
{
    if (!node || !node.IsSequence()) {
        return std::nullopt;
    }

    std::vector<T> values;
    values.reserve(node.size());
    for (const auto& item : node) {
        if constexpr (std::is_floating_point_v<T>) {
            values.emplace_back(static_cast<T>(item.as<double>()));
        } else {
            values.emplace_back(item.as<T>());
        }
    }
    return values;
}

template <typename Derived>
inline bool AssignEigenFromSequence(const YAML::Node& node, Derived& target)
{
    if (!node || !node.IsSequence()) {
        return false;
    }

    using Scalar = typename Derived::Scalar;
    const std::size_t expected = static_cast<std::size_t>(target.rows()) * static_cast<std::size_t>(target.cols());
    if (expected == 0 || node.size() != expected) {
        return false;
    }

    for (std::size_t idx = 0; idx < expected; ++idx) {
        const auto value = static_cast<Scalar>(node[idx].as<double>());
        const auto row = static_cast<Eigen::Index>(idx / target.cols());
        const auto col = static_cast<Eigen::Index>(idx % target.cols());
        target(row, col) = value;
    }
    return true;
}

inline void CollectImageInfo(const YAML::Node& common_node, const YAML::Node& camera_node, slam_core::CommonParams& common, slam_core::CameraParams& camera)
{
    std::vector<std::pair<int, std::string>> ordered_topics;
    common.img_topics.clear();
    for (const auto& it : common_node) {
        if (!it.first.IsScalar() || !it.second.IsScalar()) {
            continue;
        }

        const auto key = it.first.as<std::string>();
        constexpr std::string_view prefix = "img_topic_";
        if (key.rfind(prefix, 0) != 0) {
            continue;
        }

        const auto index_str = key.substr(prefix.size());
        const int index = std::stoi(index_str);
        ordered_topics.emplace_back(index, it.second.as<std::string>());
    }

    if (!ordered_topics.empty()) {
        std::sort(ordered_topics.begin(), ordered_topics.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        std::vector<std::string> topics;
        topics.reserve(ordered_topics.size());
        for (const auto& [_, topic] : ordered_topics) {
            topics.emplace_back(topic);
        }

        common.img_topics = std::move(topics);
    }

    if (!camera_node) {
        return;
    }

    int camera_count = camera.camera_num;
    if (camera_count <= 0 && camera_node) {
        if (const auto num_node = camera_node["camera_num"]; num_node && num_node.IsScalar()) {
            camera_count = num_node.as<int>();
            camera.camera_num = camera_count;
        }
    }
    camera.extrinsic_mats.clear();
    camera.rotate_mats.clear();
    camera.transform_vecs.clear();
    camera.camera_mats.clear();
    camera.dist_coeffs.clear();
    camera.img_sizes.clear();
    if (camera_count <= 0) {
        return;
    }

    camera.extrinsic_mats.reserve(camera_count);
    camera.rotate_mats.reserve(camera_count);
    camera.transform_vecs.reserve(camera_count);
    camera.camera_mats.reserve(camera_count);
    camera.dist_coeffs.reserve(camera_count);
    camera.img_sizes.reserve(camera_count);

    for (int idx = 0; idx < camera_count; ++idx) {
        const std::string suffix = std::to_string(idx);

        Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
        if (const auto extrinsic_node = camera_node["extrinsic_mat_" + suffix]) {
            AssignEigenFromSequence(extrinsic_node, extrinsic);
        }
        camera.extrinsic_mats.emplace_back(extrinsic);
        camera.rotate_mats.emplace_back(extrinsic.block<3, 3>(0, 0));
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        translation = extrinsic.block<3, 1>(0, 3);
        camera.transform_vecs.emplace_back(translation);

        Eigen::Matrix3d cam_mat = Eigen::Matrix3d::Identity();
        if (const auto cam_node = camera_node["camera_mat_" + suffix]) {
            AssignEigenFromSequence(cam_node, cam_mat);
        }
        camera.camera_mats.emplace_back(cam_mat);

        const int dist_size = camera.fisheye_en ? 4 : 5;
        Eigen::VectorXd dist_coeffs(dist_size);
        dist_coeffs.setZero();
        if (const auto dist_node = camera_node["dist_coeff_" + suffix]; dist_node && dist_node.IsSequence()) {
            const auto limit = std::min<int>(dist_size, static_cast<int>(dist_node.size()));
            for (int i = 0; i < limit; ++i) {
                dist_coeffs(i) = dist_node[i].as<double>();
            }
        }
        camera.dist_coeffs.emplace_back(dist_coeffs);

        Eigen::Vector2d img_size = Eigen::Vector2d::Zero();
        if (const auto size_node = camera_node["image_size_" + suffix]; size_node && size_node.IsSequence() && size_node.size() >= 2) {
            img_size.x() = size_node[0].as<double>();
            img_size.y() = size_node[1].as<double>();
        }
        camera.img_sizes.emplace_back(img_size);
    }
}

inline void FillCommonParams(const YAML::Node& root, slam_core::CommonParams& common, slam_core::CameraParams& camera)
{
    if (const auto common_node = root["common"]) {
        AssignIfPresent(common_node, "lid_topic", common.lid_topic);
        AssignIfPresent(common_node, "imu_topic", common.imu_topic);
        AssignIfPresent(common_node, "visloop_topic", common.visloop_topic);
        AssignIfPresent(common_node, "rtk_topic", common.rtk_topic);
        AssignIfPresent(common_node, "render_en", common.render_en);
        AssignIfPresent(common_node, "dense_pc", common.dense_pc);
        AssignIfPresent(common_node, "space_down_sample", common.space_down_sample);
        AssignIfPresent(common_node, "time_lag_imu_to_lidar", common.time_lag_imu_to_lidar);
        AssignIfPresent(common_node, "time_lag_img_to_lidar", common.time_lag_img_to_lidar);

        AssignIfPresent(common_node, "fisheye_en", camera.fisheye_en);
        AssignIfPresent(common_node, "camera_num", camera.camera_num);
        AssignIfPresent(common_node, "image_save_path", camera.image_save_path);

        CollectImageInfo(common_node, root["camera"], common, camera);
    }

    if (const auto preprocess_node = root["preprocess"]) {
        AssignIfPresent(preprocess_node, "lidar_type", common.lidar_type);
    }

    if (const auto publish_node = root["publish"]) {
        AssignIfPresent(publish_node, "path_en", common.path_en);
        AssignIfPresent(publish_node, "scan_publish_en", common.scan_pub_en);
        AssignIfPresent(publish_node, "scan_bodyframe_pub_en", common.scan_body_pub_en);
        AssignIfPresent(publish_node, "opt_map_pub_en", common.opt_map_pub_en);
    }

    if (const auto pcd_save_node = root["pcd_save"]) {
        AssignIfPresent(pcd_save_node, "pcd_save_en", common.pcd_save_en);
    }
}

inline void FillMappingParams(const YAML::Node& node, slam_core::MappingParams& mapping)
{
    if (!node) {
        return;
    }

    AssignIfPresent(node, "start_in_aggressive_motion", mapping.non_station_start);
    AssignIfPresent(node, "extrinsic_est_en", mapping.extrinsic_est_en);
    AssignIfPresent(node, "gravity_align", mapping.gravity_align);
    AssignIfPresent(node, "prop_at_freq_of_imu", mapping.prop_at_freq_of_imu);
    AssignIfPresent(node, "check_satu", mapping.check_satu);
    AssignIfPresent(node, "imu_time_inte", mapping.imu_time_inte);
    AssignIfPresent(node, "satu_acc", mapping.satu_acc);
    AssignIfPresent(node, "satu_gyro", mapping.satu_gyro);
    AssignIfPresent(node, "acc_norm", mapping.acc_norm);
    AssignIfPresent(node, "lidar_meas_cov", mapping.laser_point_cov);
    AssignIfPresent(node, "imu_meas_acc_cov", mapping.imu_meas_acc_cov);
    AssignIfPresent(node, "imu_meas_omg_cov", mapping.imu_meas_omg_cov);
    AssignIfPresent(node, "acc_cov", mapping.acc_cov);
    AssignIfPresent(node, "gyr_cov", mapping.gyr_cov);
    AssignIfPresent(node, "b_acc_cov", mapping.b_acc_cov);
    AssignIfPresent(node, "b_gyr_cov", mapping.b_gyr_cov);
    AssignIfPresent(node, "plane_thr", mapping.plane_thr);
    AssignIfPresent(node, "match_s", mapping.match_s);
    AssignIfPresent(node, "fov_degree", mapping.fov_deg);
    AssignIfPresent(node, "det_range", mapping.DET_RANGE);
    AssignIfPresent(node, "keyframe_adding_dist_thres", mapping.keyframe_adding_dist_thres);
    AssignIfPresent(node, "keyframe_adding_ang_thres", mapping.keyframe_adding_ang_thres);
    AssignIfPresent(node, "filter_size_surf_min", mapping.filter_size_surf_min);
    AssignIfPresent(node, "filter_size_map_min", mapping.filter_size_map_min);
    AssignIfPresent(node, "cube_len", mapping.cube_len);
    AssignIfPresent(node, "vel_cov", mapping.vel_cov);
    AssignIfPresent(node, "save_in_advance", mapping.save_in_advance);

    AssignEigenFromSequence(node["gravity"], mapping.gravity);
    AssignEigenFromSequence(node["gravity_init"], mapping.gravity_init);
    AssignEigenFromSequence(node["extrinsic_T"], mapping.extrinT);
    AssignEigenFromSequence(node["extrinsic_R"], mapping.extrinR);
}

inline void FillVoxelParams(const YAML::Node& node, slam_core::VoxelParams& voxel)
{
    if (!node) {
        return;
    }

    AssignIfPresent(node, "voxel_size", voxel.voxel_size);
    AssignIfPresent(node, "plannar_threshold", voxel.plannar_threshold);
    if (const auto layer_sizes = SequenceToVector<int>(node["layer_point_size"])) {
        voxel.layer_point_size = *layer_sizes;
    }
    AssignIfPresent(node, "ranging_cov", voxel.ranging_cov);
    AssignIfPresent(node, "angle_cov", voxel.angle_cov);
    AssignIfPresent(node, "acc_cov", voxel.acc_cov);
    AssignIfPresent(node, "gyr_cov", voxel.gyr_cov);
    AssignIfPresent(node, "b_acc_cov", voxel.b_acc_cov);
    AssignIfPresent(node, "b_gyr_cov", voxel.b_gyr_cov);
    AssignIfPresent(node, "max_layer", voxel.max_layer);
    AssignIfPresent(node, "max_points_size", voxel.max_points_size);
    AssignIfPresent(node, "max_cov_points_size", voxel.max_cov_points_size);
    AssignIfPresent(node, "max_iteration", voxel.max_iteration);
}

inline void FillCameraParams(const YAML::Node& node, slam_core::CameraParams& camera)
{
    if (!node) {
        return;
    }

    AssignIfPresent(node, "fisheye_en", camera.fisheye_en);
    AssignIfPresent(node, "camera_num", camera.camera_num);
    AssignIfPresent(node, "image_save_path", camera.image_save_path);
}

template <typename Derived>
inline std::string FormatEigenMatrix(const Eigen::MatrixBase<Derived>& mat)
{
    std::ostringstream oss;
    oss << "\n";
    oss << mat.format(Eigen::IOFormat(
        Eigen::StreamPrecision,  // 浮点精度
        Eigen::DontAlignCols,    // 禁用对齐
        ", ",                    // 元素分隔符
        ";\n",                   // 行分隔符
        "[",                     // 前缀
        "]",                     // 后缀
        "",                      // 行前缀
        ""));                    // 行后缀
    return oss.str();
}

}  // namespace detail

inline bool LoadConfigFromFile(const std::string& yaml_path)
{
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception&) {
        spdlog::critical("Failed to load config file: {}", yaml_path);
        return false;
    }

    auto& config = slam_core::Config::GetInstance();

    config.common_params = {};
    config.mapping_params = {};
    config.voxel_params = {};
    config.camera_params = {};

    detail::FillCommonParams(root, config.common_params, config.camera_params);
    detail::FillMappingParams(root["mapping"], config.mapping_params);
    detail::FillVoxelParams(root["voxel"], config.voxel_params);
    detail::FillCameraParams(root["camera"], config.camera_params);

    return true;
}

inline void LogConfig()
{
    const auto& config = slam_core::Config::GetInstance();
    const auto& common = config.common_params;
    spdlog::info("[Config] Common.lid_topic: {}", common.lid_topic);
    spdlog::info("[Config] Common.imu_topic: {}", common.imu_topic);
    spdlog::info("[Config] Common.visloop_topic: {}", common.visloop_topic);
    spdlog::info("[Config] Common.rtk_topic: {}", common.rtk_topic);
    spdlog::info("[Config] Common.img_topics=[{}]", fmt::join(common.img_topics, ", "));
    spdlog::info("[Config] Common.space_down_sample: {}", common.space_down_sample);
    spdlog::info("[Config] Common.time_lag_imu_to_lidar: {}", common.time_lag_imu_to_lidar);
    spdlog::info("[Config] Common.time_lag_img_to_lidar: {}", common.time_lag_img_to_lidar);
    spdlog::info("[Config] Common.pcd_save_en: {}", common.pcd_save_en);
    spdlog::info("[Config] Common.path_en: {}", common.path_en);
    spdlog::info("[Config] Common.scan_pub_en: {}", common.scan_pub_en);
    spdlog::info("[Config] Common.scan_body_pub_en: {}", common.scan_body_pub_en);
    spdlog::info("[Config] Common.opt_map_pub_en: {}", common.opt_map_pub_en);
    spdlog::info("[Config] Common.lidar_type: {}", common.lidar_type);
    spdlog::info("[Config] Common.render_en: {}", common.render_en);
    spdlog::info("[Config] Common.dense_pc: {}", common.dense_pc);

    const auto& mapping = config.mapping_params;
    spdlog::info("[Config] Mapping.non_station_start: {}", mapping.non_station_start);
    spdlog::info("[Config] Mapping.extrinsic_est_en: {}", mapping.extrinsic_est_en);
    spdlog::info("[Config] Mapping.gravity_align: {}", mapping.gravity_align);
    spdlog::info("[Config] Mapping.prop_at_freq_of_imu: {}", mapping.prop_at_freq_of_imu);
    spdlog::info("[Config] Mapping.check_satu: {}", mapping.check_satu);
    spdlog::info("[Config] Mapping.init_map_size: {}", mapping.init_map_size);
    spdlog::info("[Config] Mapping.imu_time_inte: {}", mapping.imu_time_inte);
    spdlog::info("[Config] Mapping.satu_acc: {}", mapping.satu_acc);
    spdlog::info("[Config] Mapping.satu_gyro: {}", mapping.satu_gyro);
    spdlog::info("[Config] Mapping.acc_norm: {}", mapping.acc_norm);
    spdlog::info("[Config] Mapping.laser_point_cov: {}", mapping.laser_point_cov);
    spdlog::info("[Config] Mapping.b_acc_cov: {}", mapping.b_acc_cov);
    spdlog::info("[Config] Mapping.b_gyr_cov: {}", mapping.b_gyr_cov);
    spdlog::info("[Config] Mapping.imu_meas_acc_cov: {}", mapping.imu_meas_acc_cov);
    spdlog::info("[Config] Mapping.imu_meas_omg_cov: {}", mapping.imu_meas_omg_cov);
    spdlog::info("[Config] Mapping.acc_cov: {}", mapping.acc_cov);
    spdlog::info("[Config] Mapping.gyr_cov: {}", mapping.gyr_cov);
    spdlog::info("[Config] Mapping.plane_thr: {}", mapping.plane_thr);
    spdlog::info("[Config] Mapping.match_s: {}", mapping.match_s);
    spdlog::info("[Config] Mapping.fov_deg: {}", mapping.fov_deg);
    spdlog::info("[Config] Mapping.DET_RANGE: {}", mapping.DET_RANGE);
    spdlog::info("[Config] Mapping.gravity: {}", detail::FormatEigenMatrix(mapping.gravity));
    spdlog::info("[Config] Mapping.gravity_init: {}", detail::FormatEigenMatrix(mapping.gravity_init));
    spdlog::info("[Config] Mapping.extrinT: {}", detail::FormatEigenMatrix(mapping.extrinT));
    spdlog::info("[Config] Mapping.extrinR: {}", detail::FormatEigenMatrix(mapping.extrinR));
    spdlog::info("[Config] Mapping.keyframe_adding_dist_thres: {}", mapping.keyframe_adding_dist_thres);
    spdlog::info("[Config] Mapping.keyframe_adding_ang_thres: {}", mapping.keyframe_adding_ang_thres);
    spdlog::info("[Config] Mapping.filter_size_surf_min: {}", mapping.filter_size_surf_min);
    spdlog::info("[Config] Mapping.filter_size_map_min: {}", mapping.filter_size_map_min);
    spdlog::info("[Config] Mapping.cube_len: {}", mapping.cube_len);
    spdlog::info("[Config] Mapping.vel_cov: {}", mapping.vel_cov);
    spdlog::info("[Config] Mapping.save_in_advance: {}", mapping.save_in_advance);

    const auto& voxel = config.voxel_params;
    spdlog::info("[Config] Voxel.voxel_size: {}", voxel.voxel_size);
    spdlog::info("[Config] Voxel.plannar_threshold: {}", voxel.plannar_threshold);
    spdlog::info("[Config] Voxel.layer_point_size=[{}]", fmt::join(voxel.layer_point_size, ", "));
    spdlog::info("[Config] Voxel.ranging_cov: {}", voxel.ranging_cov);
    spdlog::info("[Config] Voxel.angle_cov: {}", voxel.angle_cov);
    spdlog::info("[Config] Voxel.acc_cov: {}", voxel.acc_cov);
    spdlog::info("[Config] Voxel.gyr_cov: {}", voxel.gyr_cov);
    spdlog::info("[Config] Voxel.b_acc_cov: {}", voxel.b_acc_cov);
    spdlog::info("[Config] Voxel.b_gyr_cov: {}", voxel.b_gyr_cov);
    spdlog::info("[Config] Voxel.max_layer: {}", voxel.max_layer);
    spdlog::info("[Config] Voxel.max_points_size: {}", voxel.max_points_size);
    spdlog::info("[Config] Voxel.max_cov_points_size: {}", voxel.max_cov_points_size);
    spdlog::info("[Config] Voxel.max_iteration: {}", voxel.max_iteration);

    const auto& camera = config.camera_params;
    spdlog::info("[Config] Camera.fisheye_en: {}", camera.fisheye_en);
    spdlog::info("[Config] Camera.image_save_path: {}", camera.image_save_path);
    spdlog::info("[Config] Camera.camera_num: {}", camera.camera_num);
    for (std::size_t idx = 0; idx < static_cast<std::size_t>(camera.camera_num); ++idx) {
        spdlog::info("[Config] Camera[{}].extrinsic_mat: {}", idx, detail::FormatEigenMatrix(camera.extrinsic_mats[idx]));
        spdlog::info("[Config] Camera[{}].rotation_mat: {}", idx, detail::FormatEigenMatrix(camera.rotate_mats[idx]));
        spdlog::info("[Config] Camera[{}].transform_vec: {}", idx, detail::FormatEigenMatrix(camera.transform_vecs[idx]));
        spdlog::info("[Config] Camera[{}].camera_mat: {}", idx, detail::FormatEigenMatrix(camera.camera_mats[idx]));
        spdlog::info("[Config] Camera[{}].dist_coeff: {}", idx, detail::FormatEigenMatrix(camera.dist_coeffs[idx]));
        spdlog::info("[Config] Camera[{}].image_size: {}", idx, detail::FormatEigenMatrix(camera.img_sizes[idx]));
    }
}

}  // namespace ms_slam::slam_adapter
