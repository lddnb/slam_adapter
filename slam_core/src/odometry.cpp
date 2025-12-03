#include "slam_core/odometry.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>

#include <Eigen/Dense>
#include <easy/arbitrary_value.h>
#include <easy/profiler.h>
#include <spdlog/spdlog.h>

#include "slam_core/config.hpp"
#include "slam_core/logging_utils.hpp"

namespace ms_slam::slam_core
{

template<EstimatorConcept Estimator>
Odometry<Estimator>::Odometry()
{
    EASY_FUNCTION(profiler::colors::Amber100);
    const auto& cfg = Config::GetInstance();
    running_ = true;
    visual_enable_ = cfg.common_params.render_en;
    last_timestamp_imu_ = 0.0;
    last_index_imu_ = 0;
    odometry_thread_ = std::make_unique<std::thread>(&Odometry::RunOdometry, this);
    spdlog::info("Odometry thread initialized");
}

template<EstimatorConcept Estimator>
Odometry<Estimator>::~Odometry()
{
    EASY_FUNCTION();
    Stop();
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::Stop()
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
template<EstimatorConcept Estimator>
void Odometry<Estimator>::AddIMUData(const IMU& imu_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    if (imu_data.timestamp() <= last_timestamp_imu_) {
        spdlog::error("Current IMU data timestamp {}, last IMU data timestamp {}", imu_data.timestamp(), last_timestamp_imu_);
        spdlog::error("IMU data timestamp loop back, clear buffer");
        imu_buffer_.clear();
        last_timestamp_imu_ = 0.0;
        last_index_imu_ = 0;
        return;
    } else if (imu_data.index() - last_index_imu_ > 1) {
        spdlog::warn("IMU data lost, last index {}, current index {}", last_index_imu_, imu_data.index());
    }
    last_index_imu_ = imu_data.index();
    last_timestamp_imu_ = imu_data.timestamp();
    imu_buffer_.emplace_back(imu_data);
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::AddLidarData(const PointCloudType::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    lidar_buffer_.emplace_back(lidar_data);
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::AddImageData(const Image& image_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    image_buffer_.emplace_back(image_data);
}

template<EstimatorConcept Estimator>
std::vector<SyncData> Odometry<Estimator>::SyncPackages()
{
    EASY_FUNCTION(profiler::colors::Cyan);
    LOG_EVERY_N(info, 1000, "buff size: lidar {}, imu {}, image {}", lidar_buffer_.size(), imu_buffer_.size(), image_buffer_.size());
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

        if (imu_buffer_.size() < 10) {
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

template<EstimatorConcept Estimator>
void Odometry<Estimator>::RunOdometry()
{
    EASY_THREAD_SCOPE("OdometryThread");
    while (running_) {
        std::vector<SyncData> sync_data_list = SyncPackages();
        if (sync_data_list.empty()) {
            LOG_EVERY_N(info, 1000, "No sync data available");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        for (const auto& sync_data : sync_data_list) {
            EASY_VALUE("frame_index", static_cast<int>(estimator_.FrameIndex()));
            spdlog::info(
                "Frame [{}]: PC ts: {:.3f} --> {:.3f}, size: {}, IMU ts: {:.3f} --> {:.3f}, size: {}",
                estimator_.FrameIndex(),
                sync_data.lidar_beg_time,
                sync_data.lidar_end_time,
                sync_data.lidar_data->size(),
                sync_data.imu_data.front().timestamp(),
                sync_data.imu_data.back().timestamp(),
                sync_data.imu_data.size());
            if (visual_enable_) {
                spdlog::info("Synchronized images ts: {:.3f}", sync_data_list.front().image_data.timestamp());
            }

            estimator_.ProcessSyncData(sync_data);
            if (!estimator_.IsInitialized()) {
                spdlog::warn("Estimator is initializing, skip update at pc ts {:.3f}", sync_data.lidar_beg_time);
                continue;
            }
            const auto state_snapshot = estimator_.GetStateSnapshot();
            spdlog::info(
                "[state] update pos: {:.6f} {:.6f} {:.6f}, quat: {:.6f} {:.6f} {:.6f} {:.6f}",
                state_snapshot.p().x(),
                state_snapshot.p().y(),
                state_snapshot.p().z(),
                state_snapshot.quat().x(),
                state_snapshot.quat().y(),
                state_snapshot.quat().z(),
                state_snapshot.quat().w());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::GetLidarState(typename Estimator::StatesType& buffer)
{
    estimator_.ExportLidarStates(buffer);
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer)
{
    estimator_.ExportMapCloud(cloud_buffer);
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map)
{
    EASY_FUNCTION();
    std::unique_ptr<typename Estimator::LocalMapType> placeholder;
    estimator_.ExportLocalMap(placeholder);
    if (local_map) {
        local_map->clear();
    }
}

#ifdef USE_PCL
template<EstimatorConcept Estimator>
void Odometry<Estimator>::PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    pcl_lidar_buffer_.emplace_back(lidar_data);
}

template<EstimatorConcept Estimator>
void Odometry<Estimator>::GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer)
{
    estimator_.ExportPclMapCloud(cloud_buffer);
}
#endif

// 显式实例化默认滤波器版本
template class Odometry<DefaultEstimator>;

}  // namespace ms_slam::slam_core
