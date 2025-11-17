#include "slam_core/odometry.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <limits>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <easy/profiler.h>
#include <easy/arbitrary_value.h>
#include "slam_core/config.hpp"
#include "slam_core/filter.hpp"
#include "slam_core/utils.hpp"
#include "slam_core/logging_utils.hpp"

namespace ms_slam::slam_core
{

Odometry::Odometry()
{
    EASY_FUNCTION(profiler::colors::Amber100);
    const auto& cfg = Config::GetInstance();
    running_ = true;
    visual_enable_ = cfg.common_params.render_en;
    initialized_ = false;
    last_timestamp_imu_ = 0.0;
#ifdef USE_IKDTREE
    local_map_ = std::make_unique<ikdtreeNS::KD_TREE<ikdtreeNS::ikdTree_PointType>>();
    local_map_->set_downsample_param(0.5);
#elif defined(USE_VDB)
    local_map_ = std::make_unique<VDBMap>(0.5, 100, 10);
#elif defined(USE_HASHMAP)
    local_map_ = std::make_unique<voxelHashMap>();
#elif defined(USE_VOXELMAP)
    local_map_ = std::make_unique<VoxelMap>();
#endif
    deskewed_cloud_ = std::make_shared<PointCloudType>();
    downsampled_cloud_ = std::make_shared<PointCloudType>();
    odometry_thread_ = std::make_unique<std::thread>(&Odometry::RunOdometry, this);
    state_ = State();
    state_.AddHModel("lidar", std::bind(&Odometry::ObsModel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#ifdef USE_VOXELMAP
    state_.AddHModel("voxelmap", std::bind(&Odometry::VoxelMapObsModel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif

    lidar_measurement_cov_ = std::max(static_cast<double>(cfg.mapping_params.laser_point_cov), 1e-6);

    T_i_l = Eigen::Isometry3d::Identity();
    T_i_l.linear() = cfg.mapping_params.extrinR;
    T_i_l.translation() = cfg.mapping_params.extrinT;

    frame_index_ = 0;

    spdlog::info("Odometry thread initialized");
}

Odometry::~Odometry()
{
    EASY_FUNCTION();
    Stop();
}

void Odometry::Stop()
{
    EASY_FUNCTION();
    if (!running_.exchange(false)) {
        return;
    }
    if (odometry_thread_ && odometry_thread_->joinable()) {
        odometry_thread_->join();
    }
    odometry_thread_.reset();
    spdlog::info("Odometry thread stopped");
}

// TODO: 设置缓冲区
void Odometry::AddIMUData(const IMU& imu_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    if (imu_data.timestamp() <= last_timestamp_imu_) {
        spdlog::error("Current IMU data timestamp {}, last IMU data timestamp {}", imu_data.timestamp(), last_timestamp_imu_);
        spdlog::error("IMU data timestamp loop back, clear buffer");
        imu_buffer_.clear();
        last_timestamp_imu_ = 0.0;
        return;
    }
    last_timestamp_imu_ = imu_data.timestamp();
    imu_buffer_.emplace_back(imu_data);
}

void Odometry::AddLidarData(const PointCloudType::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    lidar_buffer_.emplace_back(lidar_data);
}

void Odometry::AddImageData(const Image& image_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    image_buffer_.emplace_back(image_data);
}

std::vector<SyncData> Odometry::SyncPackages()
{
    EASY_FUNCTION(profiler::colors::Cyan);
    std::vector<SyncData> sync_data_list;
    constexpr double kImageSyncTolerance = 0.02;  // 20 ms 容忍度

    std::unique_lock<std::mutex> lock(data_mutex_);

    while (!lidar_buffer_.empty()) {
#ifdef USE_PCL
        if (pcl_lidar_buffer_.empty()) break;
        const auto& pcl_cloud = pcl_lidar_buffer_.front();
#endif

        const auto& lidar_cloud = lidar_buffer_.front();
        const auto timestamps = lidar_cloud->field_view<TimestampTag>();

        const double lidar_beg_time = timestamps.front();
        const double lidar_end_time = timestamps.back();

#ifdef USE_PCL
        const double pcl_beg_time = pcl_cloud->points.front().timestamp;
        const double pcl_end_time = pcl_cloud->points.back().timestamp;
        CHECK(lidar_beg_time - pcl_beg_time < 1e-6 && pcl_end_time - lidar_end_time < 1e-6);
#endif

        if (imu_buffer_.empty()) {
            break;
        }

        // 需要一个 IMU 数据点早于激光帧开始，若缺失则尝试容忍或丢弃激光帧
        const double first_imu_time = imu_buffer_.front().timestamp();
        if (first_imu_time > lidar_beg_time) {
            const double gap = first_imu_time - lidar_beg_time;
            spdlog::warn("Discard lidar frame at {:.3f}s: earliest IMU {:.3f}s, gap {:.3f}s exceeds tolerance", lidar_beg_time, first_imu_time, gap);
            lidar_buffer_.pop_front();
#ifdef USE_PCL
            pcl_lidar_buffer_.pop_front();
#endif
            continue;
        }
        if (imu_buffer_.back().timestamp() < lidar_end_time) {
            break;
        }

        SyncData sync_data;
        sync_data.lidar_data = lidar_cloud;
#ifdef USE_PCL
        sync_data.pcl_lidar_data = pcl_cloud;
#endif
        sync_data.lidar_beg_time = lidar_beg_time;
        sync_data.lidar_end_time = lidar_end_time;

        // 收集覆盖激光帧时间段的 IMU 数据，额外保留一个超出结束时间的样本用于插值
        std::size_t imu_consumed = 0;
        for (; imu_consumed < imu_buffer_.size(); ++imu_consumed) {
            const auto& imu_sample = imu_buffer_[imu_consumed];
            sync_data.imu_data.emplace_back(imu_sample);
            if (imu_sample.timestamp() >= lidar_end_time) {
                ++imu_consumed;  // 该时间戳已经覆盖结束时间，跳出循环
                break;
            }
        }

        // 图像同步：剔除过旧图像，选择时间最近且落在容忍范围内的帧
        if (visual_enable_) {
            std::size_t stale_count = 0;
            while (stale_count < image_buffer_.size() && image_buffer_[stale_count].timestamp() < lidar_beg_time - kImageSyncTolerance) {
                ++stale_count;
            }
            for (std::size_t i = 0; i < stale_count; ++i) {
                image_buffer_.pop_front();
            }

            double best_diff = std::numeric_limits<double>::max();
            std::size_t best_index = 0;
            bool has_candidate = false;

            for (std::size_t idx = 0; idx < image_buffer_.size(); ++idx) {
                const double ts = image_buffer_[idx].timestamp();
                if (ts > lidar_end_time + kImageSyncTolerance) {
                    break;
                }

                const double diff = std::abs(ts - lidar_beg_time);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_index = idx;
                    has_candidate = true;
                }
            }

            if (!has_candidate || best_diff > kImageSyncTolerance) {
                spdlog::debug(
                    "Waiting for image sync: lidar[{:.6f}, {:.6f}] best image diff {:.6f}",
                    lidar_beg_time,
                    lidar_end_time,
                    has_candidate ? best_diff : std::numeric_limits<double>::quiet_NaN());
                break;
            }

            sync_data.image_data = image_buffer_[best_index];
            for (std::size_t i = 0; i <= best_index; ++i) {
                image_buffer_.pop_front();
            }
        }

        // 保留第一个超出结束时间的样本在缓存中，便于下一帧插值
        if (imu_consumed > 2) {
            imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + static_cast<std::ptrdiff_t>(imu_consumed - 2));
        }
        lidar_buffer_.pop_front();
#ifdef USE_PCL
        pcl_lidar_buffer_.pop_front();
#endif
        sync_data_list.emplace_back(std::move(sync_data));
    }

    return sync_data_list;
}

void Odometry::Initialize(const SyncData& sync_data)
{
    EASY_FUNCTION(profiler::colors::LightBlue300);
    static int N(0);
    static Eigen::Vector3d gyro_avg(0., 0., 0.);
    static Eigen::Vector3d accel_avg(0., 0., 0.);
    static double last_imu_stamp = 0.0;

    for (size_t i = 0; i < sync_data.imu_data.size() - 1; ++i) {
        const auto& imu_data = sync_data.imu_data[i];
        if (N == 0) {
            gyro_avg += imu_data.angular_velocity();
            accel_avg += imu_data.linear_acceleration();
            N++;
        } else {
            if (imu_data.timestamp() <= last_imu_stamp) continue;
            N++;
            accel_avg += (imu_data.linear_acceleration() - accel_avg) / N;
            gyro_avg += (imu_data.angular_velocity() - gyro_avg) / N;
        }
        last_imu_stamp = imu_data.timestamp();
    }
    if (N >= 100) {
        const auto& cfg = Config::GetInstance();
        const Eigen::Vector3d gravity_world = cfg.mapping_params.gravity;
        const double gravity_norm = gravity_world.norm();

        state_.b_g(gyro_avg);

        imu_scale_factor_ = gravity_norm / accel_avg.norm();
        const Eigen::Vector3d tmp_gravity = -accel_avg * imu_scale_factor_;

        if (cfg.mapping_params.gravity_align) {
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

        // state_.b_a(bias_a);

        state_.timestamp(last_imu_stamp);
        imu_state_buffer_.emplace_back(state_);
        initialized_ = true;

        // clang-format off
        spdlog::info(
            "Initialize with {} IMU:g = [{:.6f}, {:.6f}, {:.6f}], imu_scale_factor_ = {:.6f},  b_g = [{:.6f}, {:.6f}, {:.6f}], b_a = [{:.6f}, {:.6f}, {:.6f}], timestamp = {:.3f}",
            N,
            state_.g().x(), state_.g().y(), state_.g().z(), imu_scale_factor_,
            state_.b_g().x(), state_.b_g().y(), state_.b_g().z(),
            state_.b_a().x(), state_.b_a().y(), state_.b_a().z(),
            state_.timestamp());
        // clang-format on
        spdlog::info("init cov: {}", as_eigen(state_.cov().block<6, 6>(0, 0)));
    }
    EASY_VALUE("init_imu_count", N, EASY_UNIQUE_VIN);
}

void Odometry::ProcessImuData(const SyncData& sync_data)
{
    EASY_FUNCTION(profiler::colors::Pink300);
    IMU last_imu(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0);
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    double dt = 0.0;
    State::BundleInput input;
    for (const auto& imu : sync_data.imu_data) {
        EASY_BLOCK("IntegrateImu", profiler::colors::Purple500);
        dt = imu.timestamp() - state_.timestamp();
        if (last_imu.timestamp() == 0.0 || dt <= 0.0) {
            last_imu = imu;
            continue;
        } else if (imu.timestamp() < sync_data.lidar_end_time) {
            gyro = 0.5 * (last_imu.angular_velocity() + imu.angular_velocity());
            acc = 0.5 * (last_imu.linear_acceleration() + imu.linear_acceleration());
            // 校正比例因子
            acc = acc * imu_scale_factor_;

            input = State::BundleInput{gyro, acc};
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
            acc = acc * imu_scale_factor_;
            input = State::BundleInput{gyro, acc};

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

            // spdlog::info("predict cov: {}", as_eigen(state_.cov().block<6, 6>(0, 0)));
            {
                std::unique_lock<std::mutex> lock(state_mutex_, std::try_to_lock);
                if (lock.owns_lock()) {
                    lidar_state_buffer_.emplace_back(state_);
                } else {
                    spdlog::info("Skip lidar_state_buffer push: state_mutex_ busy at {:.3f}s", state_.timestamp());
                }
            }
        }
        imu_state_buffer_.emplace_back(state_);
    }
    while (imu_state_buffer_.size() > 1 && imu_state_buffer_[1].timestamp() < sync_data.lidar_beg_time) {
        imu_state_buffer_.pop_front();
    }
}

/**
 * @brief 对输入点云进行时间去畸变处理
 * @param cloud 原始点云
 * @param state 当前里程计状态
 * @param buffer IMU预测状态缓冲区
 * @return 去畸变后的点云
 * @note 假设点时间戳严格位于缓冲区时间范围内
 */
PointCloudType::Ptr Odometry::Deskew(const PointCloudType::ConstPtr& cloud, const State& state, const States& buffer) const
{
    EASY_FUNCTION(profiler::colors::Teal300);
    PointCloudType::Ptr deskewed_cloud = std::make_shared<PointCloudType>(*cloud);
    Eigen::Isometry3f TN = (state.isometry3d() * T_i_l).cast<float>();

    std::vector<int> indices(cloud->size());
    std::iota(indices.begin(), indices.end(), 0);

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

    EASY_BLOCK("DeskewPoints", profiler::colors::BlueGrey500);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int k) {
        const auto& point_time = cloud->timestamp(k);

        // 返回数组中第一个大于或等于被查数的迭代器
        auto it = std::lower_bound(buffer.begin(), buffer.end(), point_time, [](const State& state_item, double target_time) {
            return state_item.timestamp() < target_time;
        });

        if (it == buffer.end()) {
            spdlog::error("Lower bound search failed for time {:.3f}", point_time);
            return;
        }

        const State& reference_state = (it->timestamp() == point_time) ? *it : *std::prev(it);

        auto X0 = reference_state.Predict(point_time);
        if (!X0) {
            spdlog::error("Failed to predict point at time {:.3f}", point_time);
        }

        Eigen::Isometry3f T0 = (X0.value() * T_i_l).cast<float>();

        Eigen::Vector3f p = cloud->position(k);

        p = TN.inverse() * T0 * p;

        deskewed_cloud->position(k) = p;
    });

    return deskewed_cloud;
}

#ifdef USE_PCL
void Odometry::PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    pcl_lidar_buffer_.emplace_back(lidar_data);
}

PointCloudT::Ptr Odometry::PCLDeskew(const PointCloudT::ConstPtr& cloud, const State& state, const States& buffer) const
{
    EASY_FUNCTION(profiler::colors::Teal900);
    if (!cloud) {
        spdlog::warn("PCLDeskew received null cloud");
        return PointCloudT::Ptr(new PointCloudT);
    }

    PointCloudT::Ptr deskewed_cloud(new PointCloudT);
    *deskewed_cloud = *cloud;

    if (cloud->points.empty()) {
        return deskewed_cloud;
    }

    const auto& config = Config::GetInstance();
    Eigen::Isometry3d T_i_l_local = Eigen::Isometry3d::Identity();
    T_i_l_local.linear() = config.mapping_params.extrinR;
    T_i_l_local.translation() = config.mapping_params.extrinT;
    Eigen::Isometry3f TN = (state.isometry3d() * T_i_l_local).cast<float>();

    // 守护时间范围，避免插值越界
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

    std::vector<std::size_t> indices(cloud->points.size());
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("DeskewPointsPCL", profiler::colors::BlueGrey300);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](std::size_t k) {
        const double point_time = cloud->points[k].timestamp;

        // 查找包络状态用于线性插值
        auto it = std::lower_bound(buffer.begin(), buffer.end(), point_time, [](const State& state_item, double target_time) {
            return state_item.timestamp() < target_time;
        });

        if (it == buffer.end()) {
            spdlog::error("Lower bound search failed for time {:.3f} (PCL)", point_time);
            return;
        }

        const State& reference_state = (it->timestamp() == point_time) ? *it : *std::prev(it);

        auto predicted_state = reference_state.Predict(point_time);
        if (!predicted_state) {
            spdlog::error("Failed to predict PCL point at time {:.3f}", point_time);
            return;
        }

        // 计算点云在标定框架下的去畸变坐标
        Eigen::Isometry3f T0 = (predicted_state.value() * T_i_l_local).cast<float>();

        Eigen::Vector3f p(cloud->points[k].x, cloud->points[k].y, cloud->points[k].z);
        Eigen::Vector3f deskewed = TN.inverse() * T0 * p;

        auto& dst = deskewed_cloud->points[k];
        dst.x = deskewed.x();
        dst.y = deskewed.y();
        dst.z = deskewed.z();
    });

    return deskewed_cloud;
}

void Odometry::GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer)
{
    std::unique_lock<std::mutex> lock(state_mutex_);
    cloud_buffer.clear();
    if (!pcl_map_cloud_buffer_.empty()) {
        cloud_buffer.swap(pcl_map_cloud_buffer_);
    }
}
#endif

void Odometry::GetLidarState(States& buffer)
{
    std::unique_lock<std::mutex> lock(state_mutex_);
    buffer.clear();
    if (!lidar_state_buffer_.empty()) {
        buffer.swap(lidar_state_buffer_);
    }
}

void Odometry::GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer)
{
    std::unique_lock<std::mutex> lock(state_mutex_);
    cloud_buffer.clear();
    if (!map_cloud_buffer_.empty()) {
        cloud_buffer.swap(map_cloud_buffer_);
    }
}

void Odometry::GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map)
{
    EASY_FUNCTION();
    std::unique_lock<std::mutex> lock(state_mutex_);
#ifdef USE_IKDTREE
    std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>>().swap(local_map_->PCL_Storage);
    // local_map_->flatten(local_map_->Root_Node, local_map_->PCL_Storage, ikdtreeNS::NOT_RECORD);
    local_map->clear();
    local_map->append(local_map_->PCL_Storage);
#elif defined(USE_VDB)
    // const auto points = local_map_->Pointcloud();
    local_map->clear();
    // local_map->append(points);
#elif defined(USE_HASHMAP)
    local_map->clear();
#endif
}

void Odometry::ObsModel(State::ObsH& H, State::ObsZ& z, State::NoiseDiag& noise_inv)
{
    EASY_FUNCTION(profiler::colors::Green500);
    H.resize(0, State::DoFObs);
    z.resize(0, 1);
    noise_inv.resize(0);
    if (frame_index_ == 0) return;

    Matches obs_matches;

    int N = downsampled_cloud_->size();

    std::vector<bool> chosen(N, false);
    Matches matches(N);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("matching", profiler::colors::BlueGrey500);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Eigen::Vector3d p = downsampled_cloud_->position(i).cast<double>();
        const Eigen::Vector3d g = state_.isometry3d() * T_i_l * p;  // global coords

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> neighbors;
        std::vector<float> pointSearchSqDis;
#ifdef USE_IKDTREE
        std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> cur_neighbors;
        ikdtreeNS::ikdTree_PointType curr_point(g.x(), g.y(), g.z());
        local_map_->Nearest_Search(curr_point, 5, cur_neighbors, pointSearchSqDis);
        for (auto neighbor : cur_neighbors) {
            neighbors.emplace_back(Eigen::Vector3f(neighbor.x, neighbor.y, neighbor.z));
        }
#elif defined(USE_VDB)
        local_map_->GetKNearestNeighbors(g.cast<float>(), 10, neighbors, pointSearchSqDis);
#elif defined(USE_HASHMAP)
        neighbors = searchNeighbors(*local_map_, g.cast<float>(), 1, 0.5, 10, 1, nullptr);
        pointSearchSqDis = std::vector<float>(5, 0);
#endif

        if (neighbors.size() < 5 or pointSearchSqDis.back() > 1.0) return;

        Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
        if (not EstimatePlane(p_abcd, neighbors, 0.1)) return;

        double dist = p_abcd.head<3>().dot(g) + p_abcd(3);

        float s = 1 - 0.9 * fabs(dist) / sqrt(p.norm());
        if (s > 0.9) {
            chosen[i] = true;
            matches[i] = Match(p, p_abcd, dist);
            matches[i].confidence = static_cast<double>(s);
        }
    });  // end for_each
    EASY_END_BLOCK;

#ifdef USE_PCL
    EASY_BLOCK("PCL_MATCHING", profiler::colors::BlueGrey300);
    if (pcl_downsampled_cloud_ && !pcl_downsampled_cloud_->empty()) {
        std::vector<int> pcl_indices(static_cast<int>(pcl_downsampled_cloud_->size()));
        std::iota(pcl_indices.begin(), pcl_indices.end(), 0);
        std::atomic<std::size_t> pcl_match_count{0};

        std::for_each(std::execution::par_unseq, pcl_indices.begin(), pcl_indices.end(), [&](int idx) {
            const auto& pcl_point = pcl_downsampled_cloud_->points[static_cast<std::size_t>(idx)];
            const Eigen::Vector3d p = Eigen::Vector3d(pcl_point.x, pcl_point.y, pcl_point.z);
            const Eigen::Vector3d g = state_.isometry3d() * T_i_l * p;

            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> neighbors;
            std::vector<float> pointSearchSqDis;
#ifdef USE_IKDTREE
            std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> cur_neighbors;
            ikdtreeNS::ikdTree_PointType curr_point(g.x(), g.y(), g.z());
            local_map_->Nearest_Search(curr_point, 5, cur_neighbors, pointSearchSqDis);
            for (auto neighbor : cur_neighbors) {
                neighbors.emplace_back(Eigen::Vector3f(neighbor.x, neighbor.y, neighbor.z));
            }
#endif

            // 记录PCL管线下的近邻命中情况
            if (neighbors.size() < 5 || pointSearchSqDis.empty() || pointSearchSqDis.back() > 5.0f) {
                return;
            }

            Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
            if (not EstimatePlane(p_abcd, neighbors, 0.1)) return;

            // chosen[idx] = true;
            // matches[idx] = Match(p, p_abcd);

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

    spdlog::info("osb matches size: {}", obs_matches.size());

    H = Eigen::MatrixXd::Zero(obs_matches.size() * State::DoFRes, State::DoFObs);
    z = Eigen::VectorXd::Zero(obs_matches.size() * State::DoFRes);
    noise_inv = Eigen::VectorXd::Zero(obs_matches.size() * State::DoFRes);

    indices.resize(obs_matches.size());
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("build_jacobian", profiler::colors::Lime600);
    std::atomic<double> residual_sum = 0.0;
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Match m = obs_matches[i];

        Eigen::Matrix3d J;  // Jacobian of R act.
        const Eigen::Vector3d g = state_.ori_R().act(T_i_l * m.p, J) + state_.p();
        const Eigen::Vector3d J_R = m.n.head(3).transpose() * J;  // Jacobian of rot

        //! 这里要用负的残差
        H.block<State::DoFRes, State::DoFObs>(i * State::DoFRes, 0) << m.n.head(3).transpose(), J_R.transpose();
        z.segment<State::DoFRes>(i * State::DoFRes).setConstant(-m.dist2plane);

        // noise_inv.segment<State::DoFRes>(i * State::DoFRes).setConstant(cov_inv);

        // ICP
        // H.block<State::DoFRes, State::DoFObs>(i * State::DoFRes, 0) << Eigen::Matrix3d::Identity(), J;
        // z.segment<State::DoFRes>(i * State::DoFRes) = -(g - m.n.head(3));
        residual_sum.fetch_add(fabs(z(i * State::DoFRes)), std::memory_order_relaxed);
    });  // end for_each
    const double base_cov_inv = 1.0 / lidar_measurement_cov_;
    noise_inv = State::NoiseDiag::Constant(obs_matches.size() * State::DoFRes, base_cov_inv);
    EASY_END_BLOCK;
    spdlog::info("Residual sum: {:.6f}", residual_sum.load());
}

void Odometry::VoxelMapObsModel(State::ObsH& H, State::ObsZ& z, State::NoiseDiag& noise_inv)
{
    EASY_FUNCTION(profiler::colors::Green500);
    H.resize(0, State::DoFObs);
    z.resize(0, 1);
    noise_inv.resize(0);
    if (frame_index_ == 0) return;

    int N = downsampled_cloud_->size();
    std::vector<PointWithCov> pv_list(N);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("matching", profiler::colors::BlueGrey500);
    
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        const Eigen::Vector3d p = downsampled_cloud_->position(i).cast<double>();
        const Eigen::Vector3d g = state_.isometry3d() * T_i_l * p;  // global coords

        PointWithCov pv;
        pv.body_point = p.cast<float>();
        pv.world_point = g.cast<float>();
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // Eigen::Matrix3d cov_lidar = CalcBodyCov(pv.body_point, ranging_cov, angle_cov);
        Eigen::Matrix3d cov_lidar = var_down_body_[i];
        // 将body系的var转换到world系
        pv.cov = TransformLiDARCovToWorld(pv.body_point, T_i_l, state_, cov_lidar);
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    });

    std::vector<PointToPlaneResidual> ptpl_list;
    std::vector<Eigen::Vector3d> non_match_list;
    local_map_->BuildResidualList(pv_list, ptpl_list, non_match_list);
    EASY_END_BLOCK;

    int effct_feat_num = ptpl_list.size();

    spdlog::info("osb matches size: {}", effct_feat_num);

    H = Eigen::MatrixXd::Zero(effct_feat_num * State::DoFRes, State::DoFObs);
    z = Eigen::VectorXd::Zero(effct_feat_num * State::DoFRes);
    noise_inv = Eigen::VectorXd::Zero(effct_feat_num * State::DoFRes);

    indices.resize(effct_feat_num);
    std::iota(indices.begin(), indices.end(), 0);

    EASY_BLOCK("build_jacobian", profiler::colors::Lime600);
    std::atomic<double> residual_sum = 0.0;
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
        Eigen::Vector3d point_this_be = ptpl_list[i].body_point.cast<double>();
        Eigen::Matrix3d point_be_crossmat = manif::skew(point_this_be);
        Eigen::Vector3d point_this = T_i_l * point_this_be;
        Eigen::Matrix3d point_crossmat = manif::skew(point_this);

        Eigen::Vector3d norm_vec = ptpl_list[i].normal.cast<double>();

        Eigen::Vector3d C(state_.R().transpose() * norm_vec);
        Eigen::Vector3d A(point_crossmat * C);

        // Eigen::Matrix3d J;  // Jacobian of R act.
        // const Eigen::Vector3d g = state_.ori_R().act(T_i_l * m.p, J) + state_.p();
        // const Eigen::Vector3d J_R = m.n.head(3).transpose() * J;  // Jacobian of rot

        //! 这里要用负的残差
        H.block<State::DoFRes, State::DoFObs>(i * State::DoFRes, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), A[0], A[1], A[2];
        double pd2 = norm_vec.dot(ptpl_list[i].world_point.cast<double>()) + ptpl_list[i].d;
                
        z.segment<State::DoFRes>(i * State::DoFRes).setConstant(-pd2);

        Eigen::Vector3d point_world = ptpl_list[i].world_point.cast<double>();
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center.cast<double>();
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal.cast<double>();
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // Eigen::Matrix3d cov_lidar = CalcBodyCov(ptpl_list[i].body_point, ranging_cov, angle_cov);
        Eigen::Matrix3d cov_lidar = ptpl_list[i].cov_lidar;
        Eigen::Matrix3d R_cov_Rt = state_.R() * T_i_l.rotation() * cov_lidar * T_i_l.rotation().transpose() * state_.R().transpose();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        noise_inv.segment<State::DoFRes>(i * State::DoFRes).setConstant(R_inv);

        residual_sum.fetch_add(fabs(z(i * State::DoFRes)), std::memory_order_relaxed);
    });  // end for_each
    EASY_END_BLOCK;
    spdlog::info("Residual sum: {:.6f}", residual_sum.load());
    
}

void Odometry::RunOdometry()
{
    EASY_THREAD_SCOPE("OdometryThread");
    const auto& cfg = Config::GetInstance();
    while (running_) {
        std::vector<SyncData> sync_data_list = SyncPackages();
        if (sync_data_list.empty()) {
            LOG_EVERY_N(info, 1000, "No sync data available");
        }

        for (const auto& sync_data : sync_data_list) {
            EASY_VALUE("frame_index", static_cast<int>(frame_index_));
            EASY_BLOCK("ProcessSyncData", profiler::colors::Lime500);
            spdlog::info(
                "Frame [{}]: PC start ts: {:.3f}, PC end ts: {:.3f}, IMU start ts: {:.3f}, IMU end ts: {:.3f}, size: {}",
                frame_index_,
                sync_data.lidar_beg_time,
                sync_data.lidar_end_time,
                sync_data.imu_data.front().timestamp(),
                sync_data.imu_data.back().timestamp(),
                sync_data.imu_data.size());
            if (visual_enable_) {
                spdlog::info("Synchronized images ts: {:.3f}", sync_data_list.front().image_data.timestamp());
            }

            if (!initialized_) {
                spdlog::warn("Odometry is initializing, sync data PC ts {:.3f}", sync_data.lidar_beg_time);
                Initialize(sync_data);
                continue;
            }
            spdlog::info("[Lidar] stamp: {:.3f}, size: {}", sync_data.lidar_beg_time, sync_data.lidar_data->size());
            ProcessImuData(sync_data);
            deskewed_cloud_ = Deskew(sync_data.lidar_data, state_, imu_state_buffer_);

            EASY_BLOCK("Filter", profiler::colors::Pink400);
            LidarFilterOptions options{.rate_active = true, .sampling_stride = static_cast<std::size_t>(cfg.common_params.point_filter_num)};
            deskewed_cloud_ = ApplyLidarFilters<PointType>(deskewed_cloud_, options);
            downsampled_cloud_ = VoxelGridSamplingPstl<PointType>(deskewed_cloud_, 0.5);
            EASY_END_BLOCK;
            spdlog::info("[Lidar] downsize {}", downsampled_cloud_->size());

#ifdef USE_PCL
            pcl_deskewed_cloud_ = PCLDeskew(sync_data.pcl_lidar_data, state_, imu_state_buffer_);
            pcl::VoxelGrid<PointT> voxel_grid;
            voxel_grid.setInputCloud(pcl_deskewed_cloud_);
            voxel_grid.setLeafSize(0.5f, 0.5f, 0.5f);
            pcl_downsampled_cloud_ = PointCloudT::Ptr(new PointCloudT);
            voxel_grid.filter(*pcl_downsampled_cloud_);
            const Eigen::Matrix4f TiL = T_i_l.matrix().cast<float>();
            spdlog::info("[PCL Lidar] downsize {}", pcl_downsampled_cloud_->size());
#endif

#ifdef USE_IKDTREE
            if (frame_index_ == 0) {
                if (downsampled_cloud_->size() > 5) {
                    downsampled_cloud_->transform(state_.isometry3d() * T_i_l);
                    std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> local_points;
                    auto ori_points = downsampled_cloud_->positions_vec3();
                    for (const auto point : ori_points) {
                        ikdtreeNS::ikdTree_PointType cur_point(point.x(), point.y(), point.z());
                        local_points.emplace_back(cur_point);
                    }
                    local_map_->Build(local_points);
                    spdlog::info("build local map with {} points", local_points.size());
                }
                frame_index_++;
                continue;
            }
#elif defined(USE_VOXELMAP)
            if (frame_index_ == 0) {
                std::vector<PointWithCov> pv_list;
                auto downsampled_world = downsampled_cloud_->transformed(state_.isometry3d() * T_i_l);
                const auto world_points = downsampled_world.positions_vec3();
                const auto body_points = downsampled_cloud_->positions_vec3();
                pv_list.reserve(world_points.size());

                for (size_t i = 0; i < world_points.size(); ++i) {
                    PointWithCov pv;
                    const Eigen::Vector3f body_pt = body_points[i];
                    const Eigen::Vector3f world_pt = world_points[i];
                    pv.body_point = body_pt;
                    pv.world_point = world_pt;
                    const Eigen::Matrix3d cov_lidar = CalcBodyCov(body_pt, 0.02, 0.05);
                    pv.cov = TransformLiDARCovToWorld(body_pt, T_i_l, state_, cov_lidar);
                    pv.cov_lidar = cov_lidar;
                    pv_list.emplace_back(pv);
                }

                local_map_->Build(pv_list);
                spdlog::info("build local map with {} points", pv_list.size());

                frame_index_++;
                continue;
            }

            var_down_body_.clear();
            const auto body_points = downsampled_cloud_->positions_vec3();
            var_down_body_.reserve(body_points.size());
            for (const Eigen::Vector3f& pt : body_points) {
                var_down_body_.emplace_back(CalcBodyCov(pt, 0.02, 0.05));
            }
#endif

            EASY_BLOCK("Update", profiler::colors::DeepPurpleA400);
#ifdef USE_VOXELMAP
            state_.UpdateWithModel("voxelmap");
#else
            state_.UpdateWithModel("lidar");
#endif
            EASY_END_BLOCK;
            spdlog::info(
                "[state] update pos: {:.6f} {:.6f} {:.6f}, quat: {:.6f} {:.6f} {:.6f} {:.6f}",
                state_.p().x(),
                state_.p().y(),
                state_.p().z(),
                state_.quat().x(),
                state_.quat().y(),
                state_.quat().z(),
                state_.quat().w());
            // spdlog::info("update cov: {}", as_eigen(state_.cov().block<6, 6>(0, 0)));

            EASY_BLOCK("TransformToWorld", profiler::colors::Indigo700);
#if !defined(USE_VDB) && !defined(USE_VOXELMAP)
            downsampled_cloud_->transform(state_.isometry3d() * T_i_l);
#endif
            deskewed_cloud_->transform(state_.isometry3d() * T_i_l);
            map_cloud_buffer_.emplace_back(deskewed_cloud_->clone());
            EASY_END_BLOCK;

#ifdef USE_PCL
            // 将PCL降采样结果同样映射到世界坐标系
            const Eigen::Matrix4f state_transform = (state_.isometry3d() * T_i_l).matrix().cast<float>();
            pcl::transformPointCloud(*pcl_deskewed_cloud_, *pcl_deskewed_cloud_, state_transform);
            PointCloudT::Ptr pcl_clone(new PointCloudT(*pcl_deskewed_cloud_));
            pcl_map_cloud_buffer_.emplace_back(pcl_clone);
#endif

            EASY_BLOCK("UpdateLocalMap", profiler::colors::DarkBrown);
#ifdef USE_IKDTREE
            auto ori_points = downsampled_cloud_->positions_vec3();
            constexpr int kMinMatchPoints = 5;
            constexpr int kNumMatchPoints = 10;
            const float filter_size_map_min = 0.5;
            const float half_voxel = 0.5F * filter_size_map_min;

            std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> filtered_points;
            std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> direct_points;
            filtered_points.reserve(ori_points.size());

            for (const auto& point_world : ori_points) {
                ikdtreeNS::ikdTree_PointType curr_point(point_world.x(), point_world.y(), point_world.z());

                std::vector<ikdtreeNS::ikdTree_PointType, Eigen::aligned_allocator<ikdtreeNS::ikdTree_PointType>> neighbors;
                std::vector<float> neighbor_sq_dists;
                local_map_->Nearest_Search(curr_point, kNumMatchPoints, neighbors, neighbor_sq_dists);

                const Eigen::Vector3f center = ((point_world / filter_size_map_min).array().floor() + 0.5F) * filter_size_map_min;

                if (!neighbors.empty()) {
                    const Eigen::Vector3f nearest(neighbors.front().x, neighbors.front().y, neighbors.front().z);
                    const Eigen::Vector3f delta = nearest - center;
                    if (std::abs(delta.x()) > half_voxel && std::abs(delta.y()) > half_voxel && std::abs(delta.z()) > half_voxel) {
                        direct_points.emplace_back(curr_point);
                        continue;
                    }
                }

                bool need_add = neighbors.size() < kMinMatchPoints;
                if (!need_add) {
                    const Eigen::Vector3f center = ((point_world / filter_size_map_min).array().floor() + 0.5F) * filter_size_map_min;
                    const float dist_to_center = (point_world - center).norm();
                    need_add = true;
                    for (const auto& nb : neighbors) {
                        const Eigen::Vector3f nb_vec(nb.x, nb.y, nb.z);
                        if ((nb_vec - center).norm() < dist_to_center + 1e-6F) {
                            need_add = false;
                            break;
                        }
                    }
                }

                if (need_add) {
                    filtered_points.emplace_back(curr_point);
                }
            }

            if (!filtered_points.empty()) {
                local_map_->Add_Points(filtered_points, true);
            }
            if (!direct_points.empty()) {
                local_map_->Add_Points(direct_points, false);
            }
            spdlog::info("local map add {} filtered points and {} direct points", filtered_points.size(), direct_points.size());
#elif defined(USE_VDB)
            auto ori_points = downsampled_cloud_->positions_vec3();
            std::vector<Eigen::Vector3f> local_points;
            local_points.assign(ori_points.begin(), ori_points.end());
            local_map_->Update(local_points, state_.isometry3d() * T_i_l);
            spdlog::info("local map add {} points", local_points.size());
#elif defined(USE_HASHMAP)
            auto ori_points = downsampled_cloud_->positions_vec3();
            for (const Eigen::Vector3f& point : ori_points) {
                addPointToMap(*local_map_, point, 0.5, 20, 0.05, 0);
            }
            Eigen::Vector3d location = state_.p();
            std::vector<voxel> voxels_to_erase;
            for (const auto& pair : *local_map_) {
                Eigen::Vector3f pt = pair.second.points[0];
                if ((pt.cast<double>() - location).squaredNorm() > (100 * 100)) {
                    voxels_to_erase.emplace_back(pair.first);
                }
            }
            for (auto& vox : voxels_to_erase) local_map_->erase(vox);
            std::vector<voxel>().swap(voxels_to_erase);
#elif defined(USE_VOXELMAP)
            auto downsampled_world = downsampled_cloud_->transformed(state_.isometry3d() * T_i_l);
            auto world_points = downsampled_world.positions_vec3();
            auto lidar_points = downsampled_cloud_->positions_vec3();
            std::vector<PointWithCov> pv_list;
            pv_list.reserve(world_points.size());
            for (size_t i = 0; i < world_points.size(); i++) {
                const Eigen::Vector3f& point_lidar = lidar_points[i];
                const Eigen::Vector3f& point_world = world_points[i];
                // 保存body系和world系坐标
                PointWithCov pv;
                // 计算lidar点的cov
                // FIXME 这里错误的使用世界系的点来CalcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好
                // 需要考虑是不是init_plane时使用更大的cov更好 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
                // Eigen::Matrix3d cov_lidar = CalcBodyCov(pv.body_point, ranging_cov, angle_cov);
                const Eigen::Matrix3d& cov_lidar = var_down_body_[i];
                // 将body系的var转换到world系
                pv.cov = TransformLiDARCovToWorld(point_lidar, T_i_l, state_, cov_lidar);

                // 最终更新VoxelMap需要用的是world系的point
                pv.body_point = point_lidar;
                pv.world_point = point_world;
                pv.cov_lidar = cov_lidar;
                pv_list.push_back(pv);
            }

            std::sort(pv_list.begin(), pv_list.end(), VarContrast);
            local_map_->UpdateParallel(pv_list);
            spdlog::info("local map add {} points", world_points.size());
#endif
            frame_index_++;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

}  // namespace ms_slam::slam_core
