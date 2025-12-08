#include "slam_core/filter_odom.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <numeric>
#include <type_traits>

#include <easy/arbitrary_value.h>
#include <easy/profiler.h>

#include "slam_core/logging_utils.hpp"
#include "slam_core/localmap_traits.hpp"

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
    state_.AddHModel(
        "lidar",
        std::bind(&FilterOdom<LocalMap>::ObsModel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    spdlog::info("FilterOdom constructed");
}

template <typename LocalMap>
typename FilterOdom<LocalMap>::StateType FilterOdom<LocalMap>::GetStateSnapshot() const
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    return state_;
}

template <typename LocalMap>
CommonState FilterOdom<LocalMap>::GetState()  const
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    CommonState view{};
    view.p(state_.p());
    view.quat(state_.quat());
    view.v(state_.v());
    view.b_g(state_.b_g());
    view.b_a(state_.b_a());
    view.g(state_.g());
    view.timestamp(state_.timestamp());
    Eigen::Matrix<double, CommonState::DoF, CommonState::DoF> cov = state_.cov().block<CommonState::DoF, CommonState::DoF>(0, 0);
    view.cov(cov);
    return view;
}

template <typename LocalMap>
void FilterOdom<LocalMap>::ExportLidarStates(std::vector<CommonState>& out)
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    out.clear();
    if (!lidar_state_buffer_.empty()) {
        out.swap(lidar_state_buffer_);
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
    this->downsampled_cloud_ = VoxelGridSamplingPstl<PointType>(this->deskewed_cloud_, cfg_.mapping_params.down_size);
    EASY_END_BLOCK;
    spdlog::info("[Lidar] downsize {}", this->downsampled_cloud_->size());

#ifdef USE_PCL
    this->pcl_deskewed_cloud_ = PCLDeskew(sync_data.pcl_lidar_data);
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(this->pcl_deskewed_cloud_);
    voxel_grid.setLeafSize(0.5f, 0.5f, 0.5f);
    this->pcl_downsampled_cloud_ = PointCloudT::Ptr(new PointCloudT);
    voxel_grid.filter(*this->pcl_downsampled_cloud_);
    spdlog::info("[PCL Lidar] downsize {}", this->pcl_downsampled_cloud_->size());
#endif

    UpdateWithModel();
    UpdateLocalMap();
}

template <typename LocalMap>
void FilterOdom<LocalMap>::PushLidarState(const StateType& state)
{
    std::unique_lock<std::mutex> lock(this->state_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
        lidar_state_buffer_.emplace_back(state.ExportCommonState());
    } else {
        spdlog::info("Skip lidar_state_buffer push: state_mutex busy at {:.3f}s", state.timestamp());
    }
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

            PushLidarState(state_);
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

#ifdef USE_PCL
template <typename LocalMap>
PointCloudT::Ptr FilterOdom<LocalMap>::PCLDeskew(const PointCloudT::ConstPtr& cloud) const
{
    EASY_FUNCTION(profiler::colors::Teal900);
    if (!cloud) {
        spdlog::warn("PCLDeskew received null cloud");
        return PointCloudT::Ptr(new PointCloudT);
    }

    if (cloud->points.empty()) {
        return PointCloudT::Ptr(new PointCloudT(*cloud));
    }

    if (cloud->points.front().timestamp < this->imu_state_buffer_.front().timestamp()) {
        spdlog::error(
            "PCL cloud timestamp is earlier than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->points.front().timestamp,
            this->imu_state_buffer_.front().timestamp());
    } else if (cloud->points.back().timestamp > this->imu_state_buffer_.back().timestamp()) {
        spdlog::error(
            "PCL cloud timestamp is later than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->points.back().timestamp,
            this->imu_state_buffer_.back().timestamp());
    }

    PoseAtTimeFn pose_query = [&](double target_time) -> std::optional<Eigen::Isometry3d> {
        auto it = std::lower_bound(
            this->imu_state_buffer_.begin(),
            this->imu_state_buffer_.end(),
            target_time,
            [](const FilterState& state_item, double t) { return state_item.timestamp() < t; });

        if (it == this->imu_state_buffer_.end()) {
            spdlog::error("Lower bound search failed for time {:.3f} (PCL)", target_time);
            return std::nullopt;
        }

        const FilterState& reference_state = (it->timestamp() == target_time) ? *it : *std::prev(it);
        auto predicted = reference_state.Predict(target_time);
        if (!predicted) {
            spdlog::error("Failed to predict PCL point at time {:.3f}", target_time);
            return std::nullopt;
        }
        return predicted.value();
    };

    return DeskewPclPointCloud(cloud, pose_query, state_.isometry3d(), this->T_i_l_);
}
#endif

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

    EASY_BLOCK("matching", profiler::colors::BlueGrey500);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Eigen::Vector3d p = this->downsampled_cloud_->position(i).template cast<double>();
        const Eigen::Vector3d g = state_.isometry3d() * this->T_i_l_ * p;

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> neighbors;
        std::vector<float> pointSearchSqDis;
        MapTraits<LocalMap>::Knn(*this->local_map_, g.cast<float>(), this->localmap_params_.knn_num, neighbors, pointSearchSqDis);

        if (neighbors.size() < this->localmap_params_.min_knn_num || pointSearchSqDis.empty() || pointSearchSqDis.back() > 1.0) return;

        Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
        if (not EstimatePlane(p_abcd, neighbors, this->localmap_params_.plane_threshold)) return;

        double dist = p_abcd.head<3>().dot(g) + p_abcd(3);

        float s = 1 - 0.9 * fabs(dist) / sqrt(p.norm());
        if (s > 0.9) {
            chosen[i] = 1;
            matches[i] = Match(p, p_abcd, dist);
            matches[i].confidence = static_cast<double>(s);
        }
    });
    EASY_END_BLOCK;

    obs_matches.clear();
    for (int i = 0; i < N; i++) {
        if (chosen[i]) obs_matches.emplace_back(matches[i]);
    }

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
    OdomBaseImpl<LocalMap>::UpdateLocalMap(world_T_lidar, state_.p(), this->deskewed_cloud_, this->downsampled_cloud_);
}

template class FilterOdom<VDBMap>;
template class FilterOdom<VoxelHashMap>;
template class FilterOdom<thuni::Octree>;
}  // namespace ms_slam::slam_core
