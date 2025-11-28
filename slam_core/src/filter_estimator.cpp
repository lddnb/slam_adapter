#include "slam_core/filter_estimator.hpp"

#include <algorithm>
#include <atomic>
#include <execution>
#include <numeric>

#include <easy/profiler.h>

#include "slam_core/config.hpp"
#include "slam_core/logging_utils.hpp"
#include "slam_core/localmap_traits.hpp"

namespace ms_slam::slam_core
{
template <typename LocalMap>
void FilterEstimator<LocalMap>::ProcessImuData(
    const SyncData& sync_data,
    FilterState& state,
    FilterStates& imu_buffer,
    FilterStates& lidar_buffer,
    double imu_scale_factor,
    std::mutex& state_mutex)
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
        dt = imu.timestamp() - state.timestamp();
        if (last_imu.timestamp() == 0.0 || dt <= 0.0) {
            last_imu = imu;
            continue;
        } else if (imu.timestamp() < sync_data.lidar_end_time) {
            gyro = 0.5 * (last_imu.angular_velocity() + imu.angular_velocity());
            acc = 0.5 * (last_imu.linear_acceleration() + imu.linear_acceleration());
            acc = acc * imu_scale_factor;

            input = FilterState::BundleInput{gyro, acc};
            state.Predict(input, dt, imu.timestamp());

            last_imu = imu;
        } else {
            dt = sync_data.lidar_end_time - state.timestamp();
            if (dt <= 0.0) {
                spdlog::error("Invalid IMU data timestamp, dt {:.3f}", dt);
                continue;
            }
            CHECK(last_imu.timestamp() == state.timestamp());
            double dt_1 = imu.timestamp() - sync_data.lidar_end_time;
            double dt_2 = sync_data.lidar_end_time - last_imu.timestamp();
            double w1 = dt_1 / (dt_1 + dt_2);
            double w2 = dt_2 / (dt_1 + dt_2);
            gyro = w1 * last_imu.angular_velocity() + w2 * imu.angular_velocity();
            acc = w1 * last_imu.linear_acceleration() + w2 * imu.linear_acceleration();
            acc = acc * imu_scale_factor;
            input = FilterState::BundleInput{gyro, acc};

            state.Predict(input, dt, sync_data.lidar_end_time);
            spdlog::info(
                "[state] predict pc ts: {:.3f}, pos: {:.6f} {:.6f} {:.6f}, quat: {:.6f} {:.6f} {:.6f} {:.6f}",
                state.timestamp(),
                state.p().x(),
                state.p().y(),
                state.p().z(),
                state.quat().x(),
                state.quat().y(),
                state.quat().z(),
                state.quat().w());

            {
                std::unique_lock<std::mutex> lock(state_mutex, std::try_to_lock);
                if (lock.owns_lock()) {
                    lidar_buffer.emplace_back(state);
                } else {
                    spdlog::info("Skip lidar_state_buffer push: state_mutex_ busy at {:.3f}s", state.timestamp());
                }
            }
        }
        imu_buffer.emplace_back(state);
    }
    while (imu_buffer.size() > 1 && imu_buffer[1].timestamp() < sync_data.lidar_beg_time) {
        imu_buffer.pop_front();
    }
}

template <typename LocalMap>
PointCloudType::Ptr FilterEstimator<LocalMap>::Deskew(
    const PointCloudType::ConstPtr& cloud,
    const FilterState& state,
    const FilterStates& buffer,
    const Eigen::Isometry3d& T_i_l) const
{
    EASY_FUNCTION(profiler::colors::Teal300);
    if (!cloud) {
        spdlog::warn("Deskew received null cloud");
        return PointCloudType::Ptr(new PointCloudType);
    }

    if (cloud->empty()) {
        return cloud->clone();
    }

    if (cloud->timestamp(0) < buffer.front().timestamp()) {
        spdlog::error(
            "cloud timestamp is earlier than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->timestamp(0),
            buffer.front().timestamp());
    } else if (cloud->timestamp(cloud->size() - 1) > buffer.back().timestamp()) {
        spdlog::error(
            "cloud timestamp is later than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->timestamp(cloud->size() - 1),
            buffer.back().timestamp());
    }

    PoseAtTimeFn pose_query = [&](double target_time) -> std::optional<Eigen::Isometry3d> {
        auto it = std::lower_bound(buffer.begin(), buffer.end(), target_time, [](const FilterState& state_item, double t) {
            return state_item.timestamp() < t;
        });

        if (it == buffer.end()) {
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

    return DeskewPointCloud(cloud, pose_query, state.isometry3d(), T_i_l);
}

#ifdef USE_PCL
template <typename LocalMap>
PointCloudT::Ptr FilterEstimator<LocalMap>::PCLDeskew(
    const PointCloudT::ConstPtr& cloud,
    const FilterState& state,
    const FilterStates& buffer,
    const Eigen::Isometry3d& T_i_l) const
{
    EASY_FUNCTION(profiler::colors::Teal900);
    if (!cloud) {
        spdlog::warn("PCLDeskew received null cloud");
        return PointCloudT::Ptr(new PointCloudT);
    }

    if (cloud->points.empty()) {
        return PointCloudT::Ptr(new PointCloudT(*cloud));
    }

    if (cloud->points.front().timestamp < buffer.front().timestamp()) {
        spdlog::error(
            "PCL cloud timestamp is earlier than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->points.front().timestamp,
            buffer.front().timestamp());
    } else if (cloud->points.back().timestamp > buffer.back().timestamp()) {
        spdlog::error(
            "PCL cloud timestamp is later than buffer timestamp, cloud ts {:.3f}, buffer ts {:.3f}",
            cloud->points.back().timestamp,
            buffer.back().timestamp());
    }

    PoseAtTimeFn pose_query = [&](double target_time) -> std::optional<Eigen::Isometry3d> {
        auto it = std::lower_bound(buffer.begin(), buffer.end(), target_time, [](const FilterState& state_item, double t) {
            return state_item.timestamp() < t;
        });

        if (it == buffer.end()) {
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

    return DeskewPclPointCloud(cloud, pose_query, state.isometry3d(), T_i_l);
}
#endif

template <typename LocalMap>
void FilterEstimator<LocalMap>::ObsModel(
    const PointCloudType::Ptr& downsampled_cloud,
#ifdef USE_PCL
    const PointCloudT::Ptr& pcl_downsampled_cloud,
#endif
    LocalMap& local_map,
    const Eigen::Isometry3d& T_i_l,
    const LocalMapParams& localmap_params,
    std::size_t frame_index,
    double lidar_measurement_cov,
    const FilterState& state,
    FilterState::ObsH& H,
    FilterState::ObsZ& z,
    FilterState::NoiseDiag& noise_inv) const
{
    EASY_FUNCTION(profiler::colors::Green500);
    H.resize(0, FilterState::DoFObs);
    z.resize(0, 1);
    noise_inv.resize(0);
    if (frame_index == 0 || !downsampled_cloud) return;

    Matches obs_matches;

    int N = downsampled_cloud->size();
    //! 警惕vector<bool>并行写入竞争
    std::vector<std::uint8_t> chosen(N, 0);
    Matches matches(N);
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("matching", profiler::colors::BlueGrey500);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Eigen::Vector3d p = downsampled_cloud->position(i).cast<double>();
        const Eigen::Vector3d g = state.isometry3d() * T_i_l * p;

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> neighbors;
        std::vector<float> pointSearchSqDis;
        MapTraits<LocalMap>::Knn(local_map, g.cast<float>(), localmap_params.knn_num, neighbors, pointSearchSqDis);

        if (neighbors.size() < localmap_params.min_knn_num || pointSearchSqDis.empty() || pointSearchSqDis.back() > 1.0) return;

        Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
        if (not EstimatePlane(p_abcd, neighbors, localmap_params.plane_threshold)) return;

        double dist = p_abcd.head<3>().dot(g) + p_abcd(3);

        float s = 1 - 0.9 * fabs(dist) / sqrt(p.norm());
        if (s > 0.9) {
            chosen[i] = 1;
            matches[i] = Match(p, p_abcd, dist);
            matches[i].confidence = static_cast<double>(s);
        }
    });
    EASY_END_BLOCK;

#ifdef USE_PCL
    EASY_BLOCK("PCL_MATCHING", profiler::colors::BlueGrey300);
    if (pcl_downsampled_cloud && !pcl_downsampled_cloud->empty()) {
        std::vector<int> pcl_indices(static_cast<int>(pcl_downsampled_cloud->size()));
        std::iota(pcl_indices.begin(), pcl_indices.end(), 0);
        std::atomic<std::size_t> pcl_match_count{0};

        std::for_each(std::execution::par_unseq, pcl_indices.begin(), pcl_indices.end(), [&](int idx) {
            const auto& pcl_point = pcl_downsampled_cloud->points[static_cast<std::size_t>(idx)];
            const Eigen::Vector3d p = Eigen::Vector3d(pcl_point.x, pcl_point.y, pcl_point.z);
            const Eigen::Vector3d g = state.isometry3d() * T_i_l * p;

            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> neighbors;
            std::vector<float> pointSearchSqDis;

            if (neighbors.size() < 5 || pointSearchSqDis.empty() || pointSearchSqDis.back() > 5.0f) {
                return;
            }

            Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
            if (not EstimatePlane(p_abcd, neighbors, 0.1)) return;

            pcl_match_count.fetch_add(1, std::memory_order_relaxed);
        });

        spdlog::info("PCL candidate matches size: {}", pcl_match_count.load());
    } else {
        spdlog::warn("PCL downsampled cloud unavailable for ObsModel");
    }
    EASY_END_BLOCK;
#endif

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
        const Eigen::Vector3d gpt = state.ori_R().act(T_i_l * m.p, J) + state.p();
        const Eigen::Vector3d J_R = m.n.head(3).transpose() * J;  // Jacobian of rot

        //! 这里要用负的残差
        H.block<FilterState::DoFRes, FilterState::DoFObs>(i * FilterState::DoFRes, 0) << m.n.head(3).transpose(), J_R.transpose();
        z.segment<FilterState::DoFRes>(i * FilterState::DoFRes).setConstant(-m.dist2plane);

        // noise_inv.segment<State::DoFRes>(i * State::DoFRes).setConstant(cov_inv);

        // ICP
        // H.block<State::DoFRes, State::DoFObs>(i * State::DoFRes, 0) << Eigen::Matrix3d::Identity(), J;
        // z.segment<State::DoFRes>(i * State::DoFRes) = -(g - m.n.head(3));

        residual_sum.fetch_add(fabs(z(i * FilterState::DoFRes)), std::memory_order_relaxed);
    });
    const double base_cov_inv = 1.0 / lidar_measurement_cov;
    noise_inv = FilterState::NoiseDiag::Constant(obs_matches.size() * FilterState::DoFRes, base_cov_inv);
    EASY_END_BLOCK;
    spdlog::debug("Avg. Residual: {:.4f}", residual_sum.load() / obs_matches.size());
}

template <typename LocalMap>
void FilterEstimator<LocalMap>::UpdateWithModel(FilterState& state, std::string_view name)
{
    state.UpdateWithModel(name);
}

template class FilterEstimator<VDBMap>;
template class FilterEstimator<VoxelHashMap>;
template class FilterEstimator<thuni::Octree>;
}  // namespace ms_slam::slam_core
