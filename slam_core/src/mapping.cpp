#include "slam_core/mapping.hpp"

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
/**
 * @brief 创建滤波里程计后端的工厂函数
 * @param type 里程计后端类型
 * @return 里程计实例
 */
std::unique_ptr<OdomBase> CreateOdomEstimator(OdomType type)
{
    switch (type) {
        case OdomType::kFilterVoxelHash:
            spdlog::info("Creating FilterOdom with VoxelHashMap");
            return std::make_unique<FilterOdom<VoxelHashMap>>();
        case OdomType::kFilterOctree:
            spdlog::info("Creating FilterOdom with Octree");
            return std::make_unique<FilterOdom<thuni::Octree>>();
        case OdomType::kFilterVdb:
        default:
            spdlog::info("Creating FilterOdom with VDBMap");
            return std::make_unique<FilterOdom<VDBMap>>();
    }
}

Mapping::Mapping(OdomType type)
    : visual_enable_(false),
      estimator_(nullptr),
      mapping_thread_(nullptr),
      last_timestamp_imu_(0.0),
      last_index_imu_(0),
      running_(true)
{
    EASY_FUNCTION(profiler::colors::Amber100);
    const auto& cfg = Config::GetInstance();
    visual_enable_ = cfg.common_params.render_en;
    estimator_ = CreateOdomEstimator(type);
    if (!estimator_) {
        spdlog::error("Failed to create odom estimator for type {}", static_cast<int>(type));
    }
    mapping_thread_ = std::make_unique<std::thread>(&Mapping::RunMapping, this);
    spdlog::info("Mapping thread initialized with odometry {}", static_cast<int>(type));
}

Mapping::~Mapping()
{
    EASY_FUNCTION();
    Stop();
}

void Mapping::Stop()
{
    EASY_FUNCTION();
    if (!running_.exchange(false)) {
        return;
    }
    if (mapping_thread_ && mapping_thread_->joinable()) {
        mapping_thread_->join();
    }
    mapping_thread_.reset();
    spdlog::info("Mapping thread stopped");
}

void Mapping::AddIMUData(const IMU& imu_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    if (imu_data.timestamp() <= last_timestamp_imu_) {
        spdlog::error("Current IMU data timestamp {}, last IMU data timestamp {}", imu_data.timestamp(), last_timestamp_imu_);
        spdlog::error("IMU data timestamp loop back, clear buffer");
        imu_buffer_.clear();
        last_timestamp_imu_ = 0.0;
        last_index_imu_ = 0;
        return;
    } else if ((last_index_imu_ > 0) && (imu_data.index() - last_index_imu_ > 1)) {
        spdlog::warn("IMU data lost, last index {}, current index {}", last_index_imu_, imu_data.index());
    }
    last_index_imu_ = imu_data.index();
    last_timestamp_imu_ = imu_data.timestamp();
    imu_buffer_.emplace_back(imu_data);
}

void Mapping::AddLidarData(const PointCloudType::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    lidar_buffer_.emplace_back(lidar_data);
}

void Mapping::AddImageData(const Image& image_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    image_buffer_.emplace_back(image_data);
}

std::vector<SyncData> Mapping::SyncPackages()
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

void Mapping::RunMapping()
{
    EASY_THREAD_SCOPE("MappingThread");
    while (running_) {
        std::vector<SyncData> sync_data_list = SyncPackages();
        if (sync_data_list.empty()) {
            LOG_EVERY_N(info, 1000, "No sync data available");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        for (const auto& sync_data : sync_data_list) {
            if (!estimator_) {
                spdlog::error("Estimator is null, skip frame");
                continue;
            }
            EASY_VALUE("frame_index", static_cast<int>(estimator_->FrameIndex()));
            spdlog::info(
                "Frame [{}]: PC ts: {:.3f} --> {:.3f}, size: {}, IMU ts: {:.3f} --> {:.3f}, size: {}",
                estimator_->FrameIndex(),
                sync_data.lidar_beg_time,
                sync_data.lidar_end_time,
                sync_data.lidar_data->size(),
                sync_data.imu_data.front().timestamp(),
                sync_data.imu_data.back().timestamp(),
                sync_data.imu_data.size());
            if (visual_enable_) {
                spdlog::info("Synchronized images ts: {:.3f}", sync_data_list.front().image_data.timestamp());
            }

            estimator_->ProcessSyncData(sync_data);
            if (!estimator_->IsInitialized()) {
                spdlog::warn("Odom estimator is initializing, skip update at pc ts {:.3f}", sync_data.lidar_beg_time);
                continue;
            }
            const auto state_view = estimator_->GetState();
            spdlog::info(
                "[state] update pos: {:.6f} {:.6f} {:.6f}, quat: {:.6f} {:.6f} {:.6f} {:.6f}",
                state_view.p().x(),
                state_view.p().y(),
                state_view.p().z(),
                state_view.quat().x(),
                state_view.quat().y(),
                state_view.quat().z(),
                state_view.quat().w());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void Mapping::GetLidarState(std::vector<CommonState>& buffer)
{
    if (estimator_) {
        estimator_->ExportLidarStates(buffer);
    } else {
        buffer.clear();
    }
}

void Mapping::GetMapCloud(std::vector<PointCloudType::Ptr>& cloud_buffer)
{
    if (estimator_) {
        estimator_->ExportMapCloud(cloud_buffer);
    }
}

void Mapping::GetLocalMap(PointCloud<PointXYZDescriptor>::Ptr& local_map)
{
    EASY_FUNCTION();
    if (estimator_) {
        estimator_->ExportLocalMap(local_map);
    } else if (local_map) {
        local_map->clear();
    }
}

#ifdef USE_PCL
void Mapping::PCLAddLidarData(const PointCloudT::ConstPtr& lidar_data)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    pcl_lidar_buffer_.emplace_back(lidar_data);
}

void Mapping::GetPCLMapCloud(std::vector<PointCloudT::Ptr>& cloud_buffer)
{
    if (estimator_) {
        estimator_->ExportPclMapCloud(cloud_buffer);
    }
}
#endif

}  // namespace ms_slam::slam_core
