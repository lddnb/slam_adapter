#include "slam_core/odometry/filter_odom.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <numeric>
#include <string>
#include <type_traits>

#include <Eigen/Eigenvalues>
#include <easy/arbitrary_value.h>
#include <easy/profiler.h>

#include "slam_core/utils/logging_utils.hpp"
#include "slam_core/map/map_traits.hpp"

namespace ms_slam::slam_core
{
template <typename LocalMap>
FilterOdom<LocalMap>::FilterOdom()
: OdomBaseImpl<LocalMap>(),
  init_imu_count_(0),
  init_gyro_avg_(Eigen::Vector3d::Zero()),
  init_accel_avg_(Eigen::Vector3d::Zero()),
  init_last_imu_stamp_(0.0),
  cfg_(Config::GetInstance())
{
#ifdef USE_VOXELMAP
    static_assert(!std::is_same_v<LocalMap, VoxelMap>, "VoxelMap path will be handled separately");
#endif
    OdomBaseImpl<LocalMap>::InitializeFromConfig(cfg_);
    state_ = StateType();
    state_.AddHModel("lidar", std::bind(&FilterOdom<LocalMap>::ObsModel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    spdlog::info("FilterOdom constructed");
}

template <typename LocalMap>
typename FilterOdom<LocalMap>::StateType FilterOdom<LocalMap>::GetStateSnapshot() const
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    return state_;
}

template <typename LocalMap>
CommonState FilterOdom<LocalMap>::GetState() const
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    return state_.ExportCommonState();
}

template <typename LocalMap>
local_mapping::OdometryOutput FilterOdom<LocalMap>::GetOdomRes() const
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    return this->odom_res;
}

template <typename LocalMap>
void FilterOdom<LocalMap>::ExportStates(std::vector<CommonState>& out)
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    out.clear();
    if (!output_state_buffer_.empty()) {
        out.swap(output_state_buffer_);
    }
}

template <typename LocalMap>
void FilterOdom<LocalMap>::ProcessSyncData(const SyncData& sync_data)
{
    EASY_FUNCTION(profiler::colors::Lime700);
    if (!this->initialized_) {
        TryInitialize(sync_data);
        return;
    }

    ProcessImuData(sync_data);

    this->deskewed_cloud_ = Deskew(sync_data.lidar_data);

    EASY_BLOCK("Filter", profiler::colors::Pink400);
    LidarFilterOptions options{.rate_active = true, .sampling_stride = static_cast<std::size_t>(cfg_.common_params.point_filter_num)};
    this->deskewed_cloud_ = ApplyLidarFilters<PointType>(this->deskewed_cloud_, options);
    frame_kdtree_.SetPoints(this->deskewed_cloud_->positions_vec3());
    frame_kdtree_.Build();
    this->downsampled_cloud_ = VoxelGridSamplingPstl<PointType>(this->deskewed_cloud_, cfg_.mapping_params.down_size);
    EASY_END_BLOCK;
    spdlog::info("[Lidar] downsize {}", this->downsampled_cloud_->size());
    EstimateNormals(this->downsampled_cloud_);

    UpdateWithModel();
    UpdateLocalMap();
    std::unique_lock<std::mutex> lock(this->state_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
        output_state_buffer_.emplace_back(state_.ExportCommonState());
    }
    this->odom_res.index = this->FrameIndex();
    this->odom_res.imu_buffer.clear();
    this->odom_res.imu_buffer.assign(sync_data.imu_data.begin() + 1, sync_data.imu_data.end() - 1);
    this->odom_res.state = state_.ExportCommonState();
    this->odom_res.orig_cloud = this->deskewed_cloud_;
    this->odom_res.cloud = this->downsampled_cloud_;
    // spdlog::info("odom_res.deskewed_cloud size: {}", this->odom_res.cloud->size());

#ifdef USE_RERUN
    if (this->rec_ && this->FrameIndex() % 10 == 0) {
        if constexpr (std::is_same_v<LocalMap, VDBMap>) {
            // 可视化局部地图点云（世界系），用于观察地图增长与裁剪效果
            this->rec_->set_time_sequence("frame", static_cast<int64_t>(this->FrameIndex()));

            const auto local_map_points = MapTraits<VDBMap>::GetPointCloud(*this->local_map_);
            if (local_map_points.empty()) {
                this->rec_->log("world/local_map", rerun::Points3D::clear_fields());
            } else {
                const std::size_t stride = 1;
                std::vector<rerun::Position3D> points;
                points.reserve((local_map_points.size() + stride - 1) / stride);
                for (std::size_t i = 0; i < local_map_points.size(); i += stride) {
                    const auto& p = local_map_points[i];
                    points.emplace_back(p.x(), p.y(), p.z());
                }

                this->rec_->log("world/local_map", rerun::Points3D(points).with_colors(rerun::Color(90, 170, 255, 90)).with_radii({0.02f}));
            }
        }
    }
#endif
}

template <typename LocalMap>
void FilterOdom<LocalMap>::TryInitialize(const SyncData& sync_data)
{
    EASY_FUNCTION(profiler::colors::LightBlue300);
    for (size_t i = 0; i + 1 < sync_data.imu_data.size(); ++i) {
        const auto& imu_data = sync_data.imu_data[i];
        if (init_imu_count_ == 0) {
            init_gyro_avg_ += imu_data.angular_velocity();
            init_accel_avg_ += imu_data.linear_acceleration();
            init_imu_count_++;
        } else {
            if (imu_data.timestamp() <= init_last_imu_stamp_) continue;
            init_imu_count_++;
            init_accel_avg_ += (imu_data.linear_acceleration() - init_accel_avg_) / static_cast<double>(init_imu_count_);
            init_gyro_avg_ += (imu_data.angular_velocity() - init_gyro_avg_) / static_cast<double>(init_imu_count_);
        }
        init_last_imu_stamp_ = imu_data.timestamp();
    }
    if (init_imu_count_ >= 100) {
        const Eigen::Vector3d gravity_world = cfg_.mapping_params.gravity;
        const double gravity_norm = gravity_world.norm();

        state_.b_g(init_gyro_avg_);

        this->imu_scale_factor_ = gravity_norm / init_accel_avg_.norm();
        const Eigen::Vector3d tmp_gravity = -init_accel_avg_ * this->imu_scale_factor_;

        if (cfg_.mapping_params.gravity_align) {
            Eigen::Matrix3d hat_grav = -manif::skew(gravity_world);

            const double ref_norm = gravity_world.norm();
            const double tmp_norm = tmp_gravity.norm();
            const double align_norm = (hat_grav * tmp_gravity).norm() / (tmp_norm * ref_norm);
            double align_cos = gravity_world.dot(tmp_gravity) / (tmp_norm * ref_norm);
            align_cos = std::clamp(align_cos, -1.0, 1.0);

            Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
            if (align_norm < 1e-6) {
                if (align_cos <= 1e-6) {
                    rot = -Eigen::Matrix3d::Identity();
                }
            } else {
                Eigen::Vector3d axis = hat_grav * tmp_gravity;
                const double axis_norm = axis.norm();
                if (axis_norm > 1e-9) {
                    axis /= axis_norm;
                    const double angle = std::acos(align_cos);
                    Eigen::AngleAxisd angle_axis(angle, axis);
                    rot = angle_axis.toRotationMatrix();
                }
            }
            Eigen::Quaterniond dq(rot);
            state_.quat(dq.normalized());
            state_.g(gravity_world);
        } else {
            state_.g(tmp_gravity);
        }

        state_.timestamp(init_last_imu_stamp_);
        this->imu_state_buffer_.emplace_back(state_);
        {
            std::lock_guard<std::mutex> lock(this->state_mutex_);
            this->initialized_ = true;
            this->frame_index_ = 0;
        }

        spdlog::info(
            "Initialize with {} IMU:g = [{:.6f}, {:.6f}, {:.6f}], imu_scale_factor = {:.6f},  b_g = [{:.6f}, {:.6f}, {:.6f}], b_a = [{:.6f}, {:.6f}, "
            "{:.6f}], timestamp = {:.3f}",
            init_imu_count_,
            state_.g().x(),
            state_.g().y(),
            state_.g().z(),
            this->imu_scale_factor_,
            state_.b_g().x(),
            state_.b_g().y(),
            state_.b_g().z(),
            state_.b_a().x(),
            state_.b_a().y(),
            state_.b_a().z(),
            state_.timestamp());
        spdlog::info("init cov: {}", as_eigen(state_.cov().template block<6, 6>(0, 0)));
    }
    EASY_VALUE("init_imu_count", init_imu_count_, EASY_UNIQUE_VIN);
}

template <typename LocalMap>
void FilterOdom<LocalMap>::ProcessImuData(const SyncData& sync_data)
{
    EASY_FUNCTION(profiler::colors::Pink300);
    if (sync_data.imu_data.size() < 10) {
        spdlog::error("!!!!! lost IMU data, size {} !!!!!", sync_data.imu_data.size());
    }
    IMU last_imu(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0);
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    double dt = 0.0;
    FilterState::BundleInput input;
    for (const auto& imu : sync_data.imu_data) {
        EASY_BLOCK("IntegrateImu", profiler::colors::Purple500);
        dt = imu.timestamp() - state_.timestamp();
        if (last_imu.timestamp() == 0.0 || dt <= 0.0) {
            last_imu = imu;
            continue;
        } else if (imu.timestamp() < sync_data.lidar_end_time) {
            gyro = 0.5 * (last_imu.angular_velocity() + imu.angular_velocity());
            acc = 0.5 * (last_imu.linear_acceleration() + imu.linear_acceleration());
            acc = acc * this->imu_scale_factor_;

            input = FilterState::BundleInput{gyro, acc};
            state_.Predict(input, dt, imu.timestamp());

            last_imu = imu;
        } else {
            dt = sync_data.lidar_end_time - state_.timestamp();
            if (dt <= 0.0) {
                spdlog::error("Invalid IMU data timestamp, dt {:.3f}", dt);
                continue;
            }
            CHECK(last_imu.timestamp() == state_.timestamp());
            double dt_1 = imu.timestamp() - sync_data.lidar_end_time;
            double dt_2 = sync_data.lidar_end_time - last_imu.timestamp();
            double w1 = dt_1 / (dt_1 + dt_2);
            double w2 = dt_2 / (dt_1 + dt_2);
            gyro = w1 * last_imu.angular_velocity() + w2 * imu.angular_velocity();
            acc = w1 * last_imu.linear_acceleration() + w2 * imu.linear_acceleration();
            acc = acc * this->imu_scale_factor_;
            input = FilterState::BundleInput{gyro, acc};

            state_.Predict(input, dt, sync_data.lidar_end_time);
            spdlog::info(
                "[state] predict pc ts: {:.3f}, pos: {:.6f} {:.6f} {:.6f}, quat: {:.6f} {:.6f} {:.6f} {:.6f}",
                state_.timestamp(),
                state_.p().x(),
                state_.p().y(),
                state_.p().z(),
                state_.quat().x(),
                state_.quat().y(),
                state_.quat().z(),
                state_.quat().w());
        }
        this->imu_state_buffer_.emplace_back(state_);
    }
    while (this->imu_state_buffer_.size() > 1 && this->imu_state_buffer_[1].timestamp() < sync_data.lidar_beg_time) {
        this->imu_state_buffer_.pop_front();
    }
}

template <typename LocalMap>
PointCloudType::Ptr FilterOdom<LocalMap>::Deskew(const PointCloudType::ConstPtr& cloud) const
{
    EASY_FUNCTION(profiler::colors::Teal300);
    if (!cloud) {
        spdlog::warn("Deskew received null cloud");
        return PointCloudType::Ptr(new PointCloudType);
    }

    if (cloud->empty()) {
        return cloud->clone();
    }

    if (cloud->timestamp(0) < this->imu_state_buffer_.front().timestamp()) {
        spdlog::error(
            "cloud timestamp is earlier than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->timestamp(0),
            this->imu_state_buffer_.front().timestamp());
    } else if (cloud->timestamp(cloud->size() - 1) > this->imu_state_buffer_.back().timestamp()) {
        spdlog::error(
            "cloud timestamp is later than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->timestamp(cloud->size() - 1),
            this->imu_state_buffer_.back().timestamp());
    }

    PoseAtTimeFn pose_query = [&](double target_time) -> std::optional<Eigen::Isometry3d> {
        auto it = std::lower_bound(
            this->imu_state_buffer_.begin(),
            this->imu_state_buffer_.end(),
            target_time,
            [](const FilterState& state_item, double t) { return state_item.timestamp() < t; });

        if (it == this->imu_state_buffer_.end()) {
            spdlog::error("Lower bound search failed for time {:.3f}", target_time);
            return std::nullopt;
        }

        const FilterState& reference_state = (it->timestamp() == target_time) ? *it : *std::prev(it);
        auto predicted = reference_state.Predict(target_time);
        if (!predicted) {
            spdlog::error("Failed to predict point at time {:.3f}", target_time);
            return std::nullopt;
        }
        return predicted.value();
    };

    return DeskewPointCloud(cloud, pose_query, state_.isometry3d(), this->T_i_l_);
}

template <typename LocalMap>
void FilterOdom<LocalMap>::EstimateNormals(PointCloudType::Ptr& cloud)
{
    EASY_FUNCTION(profiler::colors::Orange400);
    std::vector<std::uint8_t> chosen(cloud->size(), 0);

    std::vector<int> indices(cloud->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<Eigen::Vector3f> normals(cloud->size(), Eigen::Vector3f::Zero());
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        // 注意：par_unseq 下必须避免共享可变容器，否则会触发数据竞争导致崩溃
        std::vector<std::size_t> nn_idx;
        std::vector<float> nn_dist2;

        const std::size_t knn_size = frame_kdtree_.KnnSearch(cloud->position(i), 5, nn_idx, nn_dist2);
        if (knn_size < 5 || nn_dist2.empty() || nn_dist2.back() > 1.0) return;

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();

        for (const auto& idx : nn_idx) {
            const Eigen::Vector3f point = frame_kdtree_.GetPoint(idx);
            centroid += point;
            covariance += point * point.transpose();
        }
        centroid /= (float)nn_idx.size();
        covariance /= (float)nn_idx.size();
        covariance -= centroid * centroid.transpose();

        //  计算协方差矩阵的特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        Eigen::Vector3f normal(solver.eigenvectors().col(0).normalized());  //  最小特征值对应的特征向量

        // 保证法向量是指向雷达这一侧
        if (normal.dot(-centroid) < 0) {
            normal *= -1.0;
        }

        normals[i] = normal;
        chosen[i] = 1;
    });

    frame_normals_.resize(normals.size(), Eigen::Vector3f::Zero());
    std::vector<std::size_t> remove_indices;
    for (const auto& i : indices) {
        if (!chosen[i]) {
            remove_indices.emplace_back(i);
        } else {
            frame_normals_.at(i) = normals[i];
        }
    }

#ifdef USE_RERUN
    if (this->rec_) {
        // 可视化法向估计阶段被剔除的点（红色加粗），用于诊断 KNN/阈值导致的稀疏区域
        this->rec_->set_time_sequence("frame", static_cast<int64_t>(this->FrameIndex()));

        std::vector<rerun::Position3D> removed_points;
        removed_points.reserve(remove_indices.size());

        const Eigen::Isometry3d world_T_lidar = state_.isometry3d() * this->T_i_l_;
        for (const auto& idx : remove_indices) {
            const Eigen::Vector3d p_world = world_T_lidar * cloud->position(idx).template cast<double>();
            removed_points.emplace_back(static_cast<float>(p_world.x()), static_cast<float>(p_world.y()), static_cast<float>(p_world.z()));
        }

        if (!removed_points.empty()) {
            this->rec_->log(
                "world/normal_estimation/removed_points",
                rerun::Points3D(removed_points).with_colors(rerun::Color(255, 0, 0, 220)).with_radii({0.03f}));
        } else {
            this->rec_->log("world/normal_estimation/removed_points", rerun::Points3D::clear_fields());
        }
    }
#endif

    // cloud->erase(remove_indices);

    CHECK(frame_normals_.size() == cloud->size());
}

template <typename LocalMap>
void FilterOdom<LocalMap>::ObsModel(StateType::ObsH& H, StateType::ObsZ& z, StateType::NoiseDiag& noise_inv)
{
    EASY_FUNCTION(profiler::colors::Green500);
    H.resize(0, FilterState::DoFObs);
    z.resize(0, 1);
    noise_inv.resize(0);
    if (this->frame_index_ == 0 || !this->downsampled_cloud_) return;

    Matches obs_matches;

    int N = this->downsampled_cloud_->size();
    //! 警惕vector<bool>并行写入竞争
    std::vector<std::uint8_t> chosen(N, 0);
    Matches matches(N);
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

#ifdef USE_RERUN
    const bool has_rerun_rec = static_cast<bool>(this->rec_);
    const double rerun_residual_threshold = 0.05;
    bool enable_rerun_vis = false;
    std::size_t rerun_frame = 0;
    std::string rerun_prefix;

    std::vector<std::uint8_t> rerun_valid;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_plane_centers;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_plane_half_sizes;
    std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf>> rerun_plane_quats;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_line_starts;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_line_ends;
    std::vector<std::uint8_t> rerun_rejected_valid;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_plane_centers;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_plane_half_sizes;
    std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf>> rerun_rejected_plane_quats;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_plane_normal_origins;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_plane_normal_vectors;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_point_normal_origins;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rerun_rejected_point_normal_vectors;

    if (has_rerun_rec) {
        rerun_frame = this->FrameIndex();
        // 仅在每个 FrameIndex 第一次进入 ObsModel 时进行可视化：如果该 index 已经渲染过则跳过
        enable_rerun_vis = (rerun_frame != rerun_last_frame_index_);
        if (enable_rerun_vis) {
            rerun_last_frame_index_ = rerun_frame;
        }
        rerun_prefix = "world/matching/iter_0";

        if (enable_rerun_vis) {
            // 通过帧序号区分不同帧的匹配可视化
            this->rec_->set_time_sequence("frame", static_cast<int64_t>(rerun_frame));

            rerun_valid.assign(static_cast<std::size_t>(N), 0);
            rerun_plane_centers.resize(static_cast<std::size_t>(N));
            rerun_plane_half_sizes.resize(static_cast<std::size_t>(N));
            rerun_plane_quats.resize(static_cast<std::size_t>(N));
            rerun_line_starts.resize(static_cast<std::size_t>(N));
            rerun_line_ends.resize(static_cast<std::size_t>(N));
            rerun_rejected_valid.assign(static_cast<std::size_t>(N), 0);
            rerun_rejected_plane_centers.resize(static_cast<std::size_t>(N));
            rerun_rejected_plane_half_sizes.resize(static_cast<std::size_t>(N));
            rerun_rejected_plane_quats.resize(static_cast<std::size_t>(N));
            rerun_rejected_plane_normal_origins.resize(static_cast<std::size_t>(N));
            rerun_rejected_plane_normal_vectors.resize(static_cast<std::size_t>(N));
            rerun_rejected_point_normal_origins.resize(static_cast<std::size_t>(N));
            rerun_rejected_point_normal_vectors.resize(static_cast<std::size_t>(N));
        }
    }
#endif

    EASY_BLOCK("matching", profiler::colors::BlueGrey500);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Eigen::Vector3d p = this->downsampled_cloud_->position(i).template cast<double>();
        const Eigen::Vector3d g = state_.isometry3d() * this->T_i_l_ * p;

        std::vector<Eigen::Matrix<float, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>> neighbors;
        std::vector<float> pointSearchSqDis;
        if constexpr (std::is_same_v<LocalMap, VDBMap>) {
            MapTraits<LocalMap>::Knn(*this->local_map_, g.cast<float>(), this->localmap_params_.knn_num, neighbors, pointSearchSqDis);
        }

        if (neighbors.size() < this->localmap_params_.min_knn_num || pointSearchSqDis.empty() || pointSearchSqDis.back() > 1.0) return;

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pts;
        Eigen::Vector3f avg_normal = Eigen::Vector3f::Zero();

        for (const auto& neighbor : neighbors) {
            pts.emplace_back(neighbor.template head<3>());
            avg_normal += neighbor.template tail<3>();
        }

        Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
        if (not EstimatePlane(p_abcd, pts, this->localmap_params_.plane_threshold)) return;

        Eigen::Vector3f normal = p_abcd.head<3>().normalized().cast<float>();
        if (normal.dot(avg_normal) < 0) {
            normal = -normal;
            // 同时翻转平面系数，保持点面距离符号一致（不改变平面几何意义）
            p_abcd.head<3>() *= -1.0;
            p_abcd(3) *= -1.0;
        }

        Eigen::Vector3d frame_normal_world = Eigen::Vector3d::Zero();
        double normal_score = 1.0;
        if (!frame_normals_[i].isZero()) {
            frame_normal_world = state_.isometry3d().rotation() * this->T_i_l_.rotation() * frame_normals_[i].cast<double>();
            normal_score = frame_normal_world.cast<float>().dot(normal);
        }

        const double dist = p_abcd.head<3>().dot(g) + p_abcd(3);

        const float s = 1 - 0.9 * fabs(dist) / sqrt(p.norm());
        //  && normal_score > 0.5
        if (s > 0.9 && normal_score > 0.5) {
            chosen[i] = 1;
            matches[i] = Match(p, p_abcd, dist);
            matches[i].confidence = static_cast<double>(s);
            if (frame_normals_[i].isZero()) {
                const Eigen::Vector3d n_frame = this->T_i_l_.rotation().transpose() * state_.isometry3d().rotation().transpose() * normal.cast<double>();
                frame_normals_[i] = n_frame.cast<float>();
            }

#ifdef USE_RERUN
            if (enable_rerun_vis) {
                // 仅可视化“残差足够大”的匹配，以减少显示数量与渲染压力
                if (std::abs(dist) < rerun_residual_threshold) {
                    return;
                }

                // 通过邻域点协方差的特征值/特征向量构造“薄椭球”，用于表达平面拟合的局部几何
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                for (const auto& pt : pts) {
                    centroid += pt.cast<double>();
                }
                centroid /= static_cast<double>(pts.size());

                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& pt : pts) {
                    const Eigen::Vector3d d = pt.cast<double>() - centroid;
                    cov.noalias() += d * d.transpose();
                }
                cov /= static_cast<double>(pts.size());

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
                if (solver.info() != Eigen::Success) {
                    return;
                }

                Eigen::Matrix3d evecs = solver.eigenvectors();  // 列向量为特征向量（按特征值从小到大排序）
                if (evecs.determinant() < 0.0) {
                    evecs.col(2) *= -1.0;  // 纠正右手系
                }
                const Eigen::Quaterniond q_world_from_ellipsoid(evecs);

                const Eigen::Vector3d evals = solver.eigenvalues().cwiseMax(1e-10);
                constexpr float kSigmaScale = 2.0f;     // 2-sigma 椭球
                constexpr float kMinThickness = 0.02f;  // 平面“厚度”下限，避免不可见
                const float hs0 = std::max(kMinThickness, kSigmaScale * static_cast<float>(std::sqrt(evals.x())));  // 法向方向
                const float hs1 = kSigmaScale * static_cast<float>(std::sqrt(evals.y()));
                const float hs2 = kSigmaScale * static_cast<float>(std::sqrt(evals.z()));

                // 点到平面投影：用于可视化点面残差（有向箭头）
                const Eigen::Vector3d n = p_abcd.head<3>();
                const double n_norm = n.norm();
                if (n_norm <= 1e-9) {
                    return;
                }
                const Eigen::Vector3d n_unit = n / n_norm;
                const double signed_dist = dist / n_norm;
                const Eigen::Vector3d g_proj = g - signed_dist * n_unit;

                const auto idx = static_cast<std::size_t>(i);
                rerun_plane_centers[idx] = centroid.cast<float>();
                rerun_plane_half_sizes[idx] = Eigen::Vector3f(hs0, hs1, hs2);
                rerun_plane_quats[idx] = q_world_from_ellipsoid.cast<float>();
                rerun_line_starts[idx] = g.cast<float>();
                rerun_line_ends[idx] = g_proj.cast<float>();
                rerun_valid[idx] = 1;
            }
#endif
        } else if (s > 0.9 && normal_score <= 0.5) {
#ifdef USE_RERUN
            if (enable_rerun_vis) {
                // 可视化法向不一致的被舍弃匹配：平面椭球与法向对比箭头
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                for (const auto& pt : pts) {
                    centroid += pt.cast<double>();
                }
                centroid /= static_cast<double>(pts.size());

                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& pt : pts) {
                    const Eigen::Vector3d d = pt.cast<double>() - centroid;
                    cov.noalias() += d * d.transpose();
                }
                cov /= static_cast<double>(pts.size());

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
                if (solver.info() != Eigen::Success) {
                    return;
                }

                Eigen::Matrix3d evecs = solver.eigenvectors();
                if (evecs.determinant() < 0.0) {
                    evecs.col(2) *= -1.0;  // 纠正右手系
                }
                const Eigen::Quaterniond q_world_from_ellipsoid(evecs);

                const Eigen::Vector3d evals = solver.eigenvalues().cwiseMax(1e-10);
                constexpr float kSigmaScale = 2.0f;
                constexpr float kMinThickness = 0.02f;
                const float hs0 = std::max(kMinThickness, kSigmaScale * static_cast<float>(std::sqrt(evals.x())));
                const float hs1 = kSigmaScale * static_cast<float>(std::sqrt(evals.y()));
                const float hs2 = kSigmaScale * static_cast<float>(std::sqrt(evals.z()));

                const Eigen::Vector3d n = p_abcd.head<3>();
                const double n_norm = n.norm();
                if (n_norm <= 1e-9) {
                    return;
                }
                const Eigen::Vector3d n_unit = n / n_norm;
                const double signed_dist = dist / n_norm;
                const Eigen::Vector3d g_proj = g - signed_dist * n_unit;

                const double frame_norm = frame_normal_world.norm();
                if (frame_norm <= 1e-9) {
                    return;
                }
                const Eigen::Vector3d frame_unit = frame_normal_world / frame_norm;
                const float normal_arrow_len = std::max(0.1f, 0.6f * std::max(hs1, hs2));

                const auto idx = static_cast<std::size_t>(i);
                rerun_rejected_plane_centers[idx] = centroid.cast<float>();
                rerun_rejected_plane_half_sizes[idx] = Eigen::Vector3f(hs0, hs1, hs2);
                rerun_rejected_plane_quats[idx] = q_world_from_ellipsoid.cast<float>();

                const Eigen::Vector3f origin = g_proj.cast<float>();
                rerun_rejected_plane_normal_origins[idx] = origin;
                rerun_rejected_plane_normal_vectors[idx] = (n_unit * normal_arrow_len).cast<float>();
                rerun_rejected_point_normal_origins[idx] = origin;
                rerun_rejected_point_normal_vectors[idx] = (frame_unit * normal_arrow_len).cast<float>();
                rerun_rejected_valid[idx] = 1;
            }
#endif
        }
    });
    EASY_END_BLOCK;

    obs_matches.clear();
    for (int i = 0; i < N; i++) {
        if (chosen[i]) obs_matches.emplace_back(matches[i]);
    }

#ifdef USE_RERUN
    if (enable_rerun_vis) {
        // 控制可视化规模，避免单帧输出过多导致 UI 卡顿
        const std::size_t target_count = obs_matches.size();
        const std::size_t rejected_count = std::count(rerun_rejected_valid.begin(), rerun_rejected_valid.end(), static_cast<std::uint8_t>(1));
        const std::size_t rejected_target_count = rejected_count;

        std::vector<rerun::components::Translation3D> plane_centers;
        std::vector<rerun::components::HalfSize3D> plane_half_sizes;
        std::vector<rerun::components::RotationQuat> plane_quats;
        std::vector<rerun::components::Position3D> residual_arrow_origins;
        std::vector<rerun::components::Vector3D> residual_arrow_vectors;
        std::vector<rerun::components::Translation3D> rejected_plane_centers;
        std::vector<rerun::components::HalfSize3D> rejected_plane_half_sizes;
        std::vector<rerun::components::RotationQuat> rejected_plane_quats;
        std::vector<rerun::components::Position3D> rejected_plane_normal_origins;
        std::vector<rerun::components::Vector3D> rejected_plane_normal_vectors;
        std::vector<rerun::components::Position3D> rejected_point_normal_origins;
        std::vector<rerun::components::Vector3D> rejected_point_normal_vectors;

        plane_centers.reserve(target_count);
        plane_half_sizes.reserve(target_count);
        plane_quats.reserve(target_count);
        residual_arrow_origins.reserve(target_count);
        residual_arrow_vectors.reserve(target_count);
        rejected_plane_centers.reserve(rejected_target_count);
        rejected_plane_half_sizes.reserve(rejected_target_count);
        rejected_plane_quats.reserve(rejected_target_count);
        rejected_plane_normal_origins.reserve(rejected_target_count);
        rejected_plane_normal_vectors.reserve(rejected_target_count);
        rejected_point_normal_origins.reserve(rejected_target_count);
        rejected_point_normal_vectors.reserve(rejected_target_count);

        for (int i = 0; i < N && plane_centers.size() < target_count; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            if (!chosen[i] || idx >= rerun_valid.size() || !rerun_valid[idx]) {
                continue;
            }

            const Eigen::Vector3f c = rerun_plane_centers[idx];
            const Eigen::Vector3f hs = rerun_plane_half_sizes[idx];
            const Eigen::Quaternionf q = rerun_plane_quats[idx];
            const Eigen::Vector3f a = rerun_line_starts[idx];
            const Eigen::Vector3f b = rerun_line_ends[idx];

            plane_centers.emplace_back(c.x(), c.y(), c.z());
            plane_half_sizes.emplace_back(hs.x(), hs.y(), hs.z());
            plane_quats.emplace_back(rerun::datatypes::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));

            // 有向残差：从点指向其在拟合平面上的投影点
            residual_arrow_origins.emplace_back(a.x(), a.y(), a.z());
            residual_arrow_vectors.emplace_back(b.x() - a.x(), b.y() - a.y(), b.z() - a.z());
        }

        for (int i = 0; i < N && rejected_plane_centers.size() < rejected_target_count; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            if (chosen[i] || idx >= rerun_rejected_valid.size() || !rerun_rejected_valid[idx]) {
                continue;
            }

            const Eigen::Vector3f c = rerun_rejected_plane_centers[idx];
            const Eigen::Vector3f hs = rerun_rejected_plane_half_sizes[idx];
            const Eigen::Quaternionf q = rerun_rejected_plane_quats[idx];
            const Eigen::Vector3f plane_o = rerun_rejected_plane_normal_origins[idx];
            const Eigen::Vector3f plane_v = rerun_rejected_plane_normal_vectors[idx];
            const Eigen::Vector3f point_o = rerun_rejected_point_normal_origins[idx];
            const Eigen::Vector3f point_v = rerun_rejected_point_normal_vectors[idx];

            rejected_plane_centers.emplace_back(c.x(), c.y(), c.z());
            rejected_plane_half_sizes.emplace_back(hs.x(), hs.y(), hs.z());
            rejected_plane_quats.emplace_back(rerun::datatypes::Quaternion::from_xyzw(q.x(), q.y(), q.z(), q.w()));
            rejected_plane_normal_origins.emplace_back(plane_o.x(), plane_o.y(), plane_o.z());
            rejected_plane_normal_vectors.emplace_back(plane_v.x(), plane_v.y(), plane_v.z());
            rejected_point_normal_origins.emplace_back(point_o.x(), point_o.y(), point_o.z());
            rejected_point_normal_vectors.emplace_back(point_v.x(), point_v.y(), point_v.z());
        }

        const rerun::Color iter_color = rerun::Color(210, 90, 255, 220);
        const rerun::Color rejected_plane_color = rerun::Color(255, 90, 90, 220);
        const rerun::Color rejected_plane_normal_color = rerun::Color(255, 190, 80, 220);
        const rerun::Color rejected_point_normal_color = rerun::Color(80, 200, 255, 220);

        if (!plane_centers.empty()) {
            this->rec_->log(
                rerun_prefix + "/planes",
                rerun::Ellipsoids3D::from_centers_and_half_sizes(plane_centers, plane_half_sizes)
                    .with_quaternions(plane_quats)
                    .with_colors({iter_color})
                    .with_fill_mode(rerun::components::FillMode::MajorWireframe)
                    .with_line_radii({rerun::Radius(0.005f)}));

            this->rec_->log(
                rerun_prefix + "/residual_arrows",
                rerun::Arrows3D::from_vectors(residual_arrow_vectors)
                    .with_origins(residual_arrow_origins)
                    .with_colors({iter_color})
                    .with_radii({rerun::Radius(0.005f)}));
        } else {
            // 当前迭代没有有效匹配时，清空对应实体，避免残留显示
            this->rec_->log(rerun_prefix + "/planes", rerun::Ellipsoids3D::clear_fields());
            this->rec_->log(rerun_prefix + "/residual_arrows", rerun::Arrows3D::clear_fields());
        }

        if (!rejected_plane_centers.empty()) {
            this->rec_->log(
                rerun_prefix + "/rejected/planes",
                rerun::Ellipsoids3D::from_centers_and_half_sizes(rejected_plane_centers, rejected_plane_half_sizes)
                    .with_quaternions(rejected_plane_quats)
                    .with_colors({rejected_plane_color})
                    .with_fill_mode(rerun::components::FillMode::MajorWireframe)
                    .with_line_radii({rerun::Radius(0.005f)}));

            this->rec_->log(
                rerun_prefix + "/rejected/plane_normals",
                rerun::Arrows3D::from_vectors(rejected_plane_normal_vectors)
                    .with_origins(rejected_plane_normal_origins)
                    .with_colors({rejected_plane_normal_color})
                    .with_radii({rerun::Radius(0.005f)}));

            this->rec_->log(
                rerun_prefix + "/rejected/point_normals",
                rerun::Arrows3D::from_vectors(rejected_point_normal_vectors)
                    .with_origins(rejected_point_normal_origins)
                    .with_colors({rejected_point_normal_color})
                    .with_radii({rerun::Radius(0.005f)}));
        } else {
            // 当前帧没有法向不一致的匹配时，清空对应实体
            this->rec_->log(rerun_prefix + "/rejected/planes", rerun::Ellipsoids3D::clear_fields());
            this->rec_->log(rerun_prefix + "/rejected/plane_normals", rerun::Arrows3D::clear_fields());
            this->rec_->log(rerun_prefix + "/rejected/point_normals", rerun::Arrows3D::clear_fields());
        }
    }
#endif

    spdlog::debug("osb matches size: {}", obs_matches.size());

    H = Eigen::MatrixXd::Zero(obs_matches.size() * FilterState::DoFRes, FilterState::DoFObs);
    z = Eigen::VectorXd::Zero(obs_matches.size() * FilterState::DoFRes);
    noise_inv = Eigen::VectorXd::Zero(obs_matches.size() * FilterState::DoFRes);

    indices.resize(obs_matches.size());
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("build_jacobian", profiler::colors::Lime600);
    std::atomic<double> residual_sum = 0.0;
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Match m = obs_matches[i];

        Eigen::Matrix3d J;
        const Eigen::Vector3d gpt = state_.ori_R().template act(this->T_i_l_ * m.p, J) + state_.p();
        const Eigen::Vector3d J_R = m.n.head(3).transpose() * J;

        //! 这里要用负的残差
        H.block<FilterState::DoFRes, FilterState::DoFObs>(i * FilterState::DoFRes, 0) << m.n.head(3).transpose(), J_R.transpose();
        z.segment<FilterState::DoFRes>(i * FilterState::DoFRes).setConstant(-m.dist2plane);

        // noise_inv.segment<State::DoFRes>(i * State::DoFRes).setConstant(cov_inv);

        // ICP
        // H.block<State::DoFRes, State::DoFObs>(i * State::DoFRes, 0) << Eigen::Matrix3d::Identity(), J;
        // z.segment<State::DoFRes>(i * State::DoFRes) = -(g - m.n.head(3));

        residual_sum.fetch_add(fabs(z(i * FilterState::DoFRes)), std::memory_order_relaxed);
    });
    const double base_cov_inv = 1.0 / this->lidar_measurement_cov_;
    noise_inv = FilterState::NoiseDiag::Constant(obs_matches.size() * FilterState::DoFRes, base_cov_inv);
    EASY_END_BLOCK;
    spdlog::debug("Avg. Residual: {:.4f}", residual_sum.load() / obs_matches.size());
}

template <typename LocalMap>
void FilterOdom<LocalMap>::UpdateWithModel()
{
    EASY_FUNCTION(profiler::colors::DeepPurpleA400);
    state_.UpdateWithModel("lidar");
}

template <typename LocalMap>
void FilterOdom<LocalMap>::UpdateLocalMap()
{
    EASY_FUNCTION(profiler::colors::DarkBrown);
    const Eigen::Isometry3d world_T_lidar = state_.isometry3d() * this->T_i_l_;
    OdomBaseImpl<LocalMap>::UpdateLocalMap(world_T_lidar, state_.p(), this->deskewed_cloud_, this->downsampled_cloud_, frame_normals_);
}

template class FilterOdom<VDBMap>;
template class FilterOdom<VoxelHashMap>;
template class FilterOdom<thuni::Octree>;
}  // namespace ms_slam::slam_core
