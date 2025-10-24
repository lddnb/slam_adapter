#include "slam_core/odometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <spdlog/spdlog.h>
#include "slam_core/config.hpp"

namespace ms_slam::slam_core
{

Odometry::Odometry()
{
    running_ = true;
    visual_enable_ = true;
    initialized_ = false;
    last_timestamp_imu_ = 0.0;
    odometry_thread_ = std::make_unique<std::thread>(&Odometry::RunOdometry, this);
}

Odometry::~Odometry()
{
    running_ = false;
    if (odometry_thread_) {
        odometry_thread_->join();
    }
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
    std::vector<SyncData> sync_data_list;
    constexpr double kImageSyncTolerance = 0.02;  // 20 ms 容忍度

    std::unique_lock<std::mutex> lock(data_mutex_);

    while (!lidar_buffer_.empty()) {
        const auto& lidar_cloud = lidar_buffer_.front();
        const auto timestamps = lidar_cloud->field_view<TimestampTag>();

        if (timestamps.empty()) {
            lidar_buffer_.pop_front();
            continue;
        }

        const double lidar_beg_time = timestamps.front();
        const double lidar_end_time = timestamps.back();

        if (imu_buffer_.empty()) {
            break;
        }

        // 需要一个 IMU 数据点早于激光帧开始，若缺失则尝试容忍或丢弃激光帧
        const double first_imu_time = imu_buffer_.front().timestamp();
        if (first_imu_time > lidar_beg_time) {
            const double gap = first_imu_time - lidar_beg_time;
            spdlog::warn("Discard lidar frame at {:.3f}s: earliest IMU {:.3f}s, gap {:.3f}s exceeds tolerance", lidar_beg_time, first_imu_time, gap);
            lidar_buffer_.pop_front();
            continue;
        }
        if (imu_buffer_.back().timestamp() < lidar_end_time) {
            break;
        }
        // 清理已经无用且早于当前帧开始时间的 IMU 数据，但保留最后一个用于插值
        while (imu_buffer_.size() > 1 && imu_buffer_[1].timestamp() <= lidar_beg_time) {
            imu_buffer_.pop_front();
        }

        SyncData sync_data;
        sync_data.lidar_data = lidar_cloud;
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
        sync_data_list.emplace_back(std::move(sync_data));
    }

    return sync_data_list;
}

void Odometry::Initialize(const SyncData& sync_data)
{
    static int N(0);
    static Eigen::Vector3d gyro_avg(0., 0., 0.);
    static Eigen::Vector3d accel_avg(0., 0., 0.);

    for (const auto& imu_data : sync_data.imu_data) {
        if (N == 0) {
            gyro_avg += imu_data.angular_velocity();
            accel_avg += imu_data.linear_acceleration();
            N++;
        } else {
            N++;
            accel_avg += (imu_data.linear_acceleration() - accel_avg) / N;
            gyro_avg += (imu_data.angular_velocity() - gyro_avg) / N;
        }
    }
    if (N >= 100) {
        spdlog::info(
            "Final imu gyro = [{:.3f}, {:.3f}, {:.3f}], accel = [{:.3f}, {:.3f}, {:.3f}]",
            gyro_avg.x(),
            gyro_avg.y(),
            gyro_avg.z(),
            accel_avg.x(),
            accel_avg.y(),
            accel_avg.z());
        Eigen::Vector3d grav_vec = accel_avg.normalized() * 9.81;
        state_.g(-grav_vec);
        state_.b_g(gyro_avg);
        state_.b_a(accel_avg - grav_vec);
        state_.timestamp(std::prev(sync_data.imu_data.end(), 2)->timestamp());
        mean_acc_ = accel_avg;
        initialized_ = true;

        // clang-format off
        spdlog::info(
            "Initialize with {} IMU:g = [{:.3f}, {:.3f}, {:.3f}], b_g = [{:.3f}, {:.3f}, {:.3f}], b_a = [{:.3f}, {:.3f}, {:.3f}], timestamp = {:.3f}",
            N,
            state_.g().x(), state_.g().y(), state_.g().z(),
            state_.b_g().x(), state_.b_g().y(), state_.b_g().z(),
            state_.b_a().x(), state_.b_a().y(), state_.b_a().z(),
            state_.timestamp());
        // clang-format on
    }
}

void Odometry::ProcessImuData(const SyncData& sync_data)
{
    IMU last_imu(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0);
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    double dt = 0.0;
    for (const auto& imu : sync_data.imu_data) {
        if (last_imu.timestamp() == 0.0) {
            last_imu = imu;
            continue;
        } else if (imu.timestamp() < sync_data.lidar_end_time) {
            dt = imu.timestamp() - state_.timestamp();
            if (dt <= 0.0) {
                spdlog::error("Invalid IMU data timestamp, imu ts {:.3f}, state ts {:.3f}, dt {:.3f}", imu.timestamp(), state_.timestamp(), dt);
                continue;
            }
            gyro = 0.5 * (last_imu.angular_velocity() + imu.angular_velocity());
            acc = 0.5 * (last_imu.linear_acceleration() + imu.linear_acceleration());
            // 校正比例因子
            acc = acc / mean_acc_.norm() * 9.81;

            const State::BundleInput input = {gyro, acc};
            state_.Predict(input, dt, imu.timestamp());

            last_imu = imu;
        } else {
            dt = sync_data.lidar_end_time - state_.timestamp();
            if (dt <= 0.0) {
                spdlog::error("Invalid IMU data timestamp, dt {:.3f}", dt);
                continue;
            }
            double dt_1 = imu.timestamp() - sync_data.lidar_end_time;
            double dt_2 = sync_data.lidar_end_time - last_imu.timestamp();
            double w1 = dt_1 / (dt_1 + dt_2);
            double w2 = dt_2 / (dt_1 + dt_2);
            gyro = w1 * last_imu.angular_velocity() + w2 * imu.angular_velocity();
            acc = w1 * last_imu.linear_acceleration() + w2 * imu.linear_acceleration();
            acc = acc / mean_acc_.norm() * 9.81;

            const State::BundleInput input = {gyro, acc};
            state_.Predict(input, dt, sync_data.lidar_end_time);
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                lidar_state_buffer_.emplace_back(state_);
            }
        }
        imu_state_buffer_.emplace_back(state_);
    }
    while (imu_state_buffer_.size() > 1 && imu_state_buffer_[1].timestamp() < sync_data.lidar_beg_time) {
        imu_state_buffer_.pop_front();
    }
}

PointCloudType::Ptr Odometry::Deskew(const PointCloudType::ConstPtr& cloud, const State& state, const States& buffer) const
{
    PointCloudType::Ptr deskewed_cloud = std::make_shared<PointCloudType>(*cloud);

    auto binary_search = [&](const double& t) {
        int l(0), r(buffer.size() - 1);

        while (l < r) {
            int m = (l + r) / 2;
            if (buffer[m].timestamp() == t)
                return m;
            else if (t < buffer[m].timestamp())
                r = m - 1;
            else
                l = m + 1;
        }

        return l - 1 > 0 ? l - 1 : l;
    };

    const auto& config = Config::GetInstance();
    Eigen::Isometry3d T_i_l = Eigen::Isometry3d::Identity();
    T_i_l.linear() = config.mapping_params.extrinR;
    T_i_l.translation() = config.mapping_params.extrinT;
    Eigen::Isometry3f TN = (state.isometry3d() * T_i_l).cast<float>();

    std::vector<int> indices(cloud->size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](int k) {
        const auto& point_time = cloud->field_view<TimestampTag>()[k];
        int i_f = binary_search(point_time);

        auto X0 = buffer[i_f].Predict(point_time);
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

void Odometry::GetLidarState(States& buffer)
{
    std::unique_lock<std::mutex> lock(state_mutex_);
    buffer.clear();
    if (!lidar_state_buffer_.empty()) {
        buffer.swap(lidar_state_buffer_);
    }
}

void Odometry::RunOdometry()
{
    while (running_) {
        std::vector<SyncData> sync_data_list = SyncPackages();
        if (!sync_data_list.empty()) {
            spdlog::info(
                "Sync data size: {}, PC start ts: {:.3f}, PC end ts: {:.3f}, IMU start ts: {:.3f}, IMU end ts: {:.3f}",
                sync_data_list.size(),
                sync_data_list.front().lidar_beg_time,
                sync_data_list.front().lidar_end_time,
                sync_data_list.front().imu_data.front().timestamp(),
                sync_data_list.front().imu_data.back().timestamp());
            if (visual_enable_) {
                spdlog::info("Synchronized images ts: {:.3f}", sync_data_list.front().image_data.timestamp());
            }
        } else {
            spdlog::info("No sync data available");
        }

        for (const auto& sync_data : sync_data_list) {
            if (!initialized_) {
                spdlog::warn("Odometry is initializing, sync data PC ts {:.3f}", sync_data.lidar_beg_time);
                Initialize(sync_data);
                continue;
            }
            ProcessImuData(sync_data);
            // auto deskewed_cloud = Deskew(sync_data.lidar_data, state_, imu_state_buffer_);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

}  // namespace ms_slam::slam_core
