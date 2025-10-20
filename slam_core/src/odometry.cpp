#include "slam_core/odometry.hpp"

#include <spdlog/spdlog.h>

namespace ms_slam::slam_core
{

Odometry::Odometry()
{
    running_ = true;
    visual_enable_ = true;
    last_timestamp_imu_ = 0;
    odometry_thread_ = std::make_unique<std::thread>(&Odometry::RunOdometry, this);
}

Odometry::~Odometry()
{
    running_ = false;
    if (odometry_thread_) {
        odometry_thread_->join();
    }
}

//TODO: 设置缓冲区
void Odometry::AddIMUData(const IMU& imu_data)
{
    if (imu_data.timestamp() <= last_timestamp_imu_) {
        spdlog::error("IMU data timestamp loop back, clear buffer");
        data_mutex_.lock();
        imu_buffer_.clear();
        data_mutex_.unlock();
        return;
    }

    data_mutex_.lock();
    imu_buffer_.emplace_back(imu_data);
    data_mutex_.unlock();
    last_timestamp_imu_ = imu_data.timestamp();
}

void Odometry::AddLidarData(const PointCloudType::ConstPtr& lidar_data)
{
    data_mutex_.lock();
    lidar_buffer_.emplace_back(lidar_data);
    data_mutex_.unlock();
}

void Odometry::AddImageData(const Image& image_data)
{
    data_mutex_.lock();
    image_buffer_.emplace_back(image_data);
    data_mutex_.unlock();
}

std::vector<SyncData> Odometry::SyncPackages()
{
    std::vector<SyncData> sync_data_list;
    if (imu_buffer_.empty() || lidar_buffer_.empty()) {
        return sync_data_list;
    }

    SyncData sync_data;
    sync_data.lidar_data = lidar_buffer_.front();
    const std::size_t pc_size = sync_data.lidar_data->size();
    sync_data.lidar_beg_time = lidar_buffer_.front()->field_view<TimestampTag>()[0];
    // sync_data.lidar_end_time = lidar_buffer_.front().field_view<TimestampTag>()[pc_size - 1];
    sync_data.lidar_end_time = sync_data.lidar_beg_time + 0.1;

    if (last_timestamp_imu_ < sync_data.lidar_beg_time) {
        return sync_data_list;
    }

    int img_remove_cnt = 0;
    if (visual_enable_) {
        for (std::size_t i = 0; i < image_buffer_.size(); ++i) {
            if (image_buffer_[i].timestamp() < sync_data.lidar_beg_time) {
                img_remove_cnt++;
            } else if (image_buffer_[i].timestamp() - sync_data.lidar_beg_time < 1e-2) {
                sync_data.image_data = image_buffer_[i];
                img_remove_cnt++;
            } else {
                break;
            }
        }
    }

    data_mutex_.lock();
    double imu_time = imu_buffer_.front().timestamp();
    while ((!imu_buffer_.empty()) && (imu_time < sync_data.lidar_end_time)) {
        imu_time = imu_buffer_.front().timestamp();
        if (imu_time > sync_data.lidar_end_time) break;
        sync_data.imu_data.emplace_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    
    lidar_buffer_.pop_front();
    for (int i = 0; i < img_remove_cnt; ++i) {
        image_buffer_.pop_front();
    }
    data_mutex_.unlock();

    sync_data_list.emplace_back(sync_data);
    return std::move(sync_data_list);
}

void Odometry::RunOdometry()
{
    while (running_) {
        std::vector<SyncData> sync_data_list = SyncPackages();
        if (!sync_data_list.empty()) {
            spdlog::info(
                "Sync data size: {}, PC ts: {:.3f}, IMU ts: {}, Image ts: {:.3f}",
                sync_data_list.size(),
                sync_data_list.front().lidar_beg_time,
                sync_data_list.front().imu_data.size(),
                sync_data_list.front().image_data.timestamp());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

}  // namespace ms_slam::slam_core