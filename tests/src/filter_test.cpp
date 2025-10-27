#include <slam_core/ply_io.hpp>
#include <slam_core/filter.hpp>
#include <spdlog/spdlog.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/stopwatch.h>

int main()
{
    // Read PLY file
    std::string ply_file_path = "/home/ubuntu/data/test_binary.ply";
    auto point_cloud = ms_slam::slam_core::ply_io::LoadPointCloud<ms_slam::slam_core::PointXYZRGBDescriptor>(ply_file_path);

    spdlog::info("Loaded point cloud with {} points", point_cloud->size());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl_point_cloud->reserve(point_cloud->size());

    // 将 slam_core 点云转换为 PCL 点云，便于直接调用 PCL 滤波器
    for (std::size_t i = 0; i < point_cloud->size(); ++i) {
        const auto position = point_cloud->position(i);
        const auto color = point_cloud->rgb(i);
        pcl::PointXYZRGB pcl_point;
        pcl_point.x = position(0);
        pcl_point.y = position(1);
        pcl_point.z = position(2);
        pcl_point.r = static_cast<std::uint8_t>(color(0));
        pcl_point.g = static_cast<std::uint8_t>(color(1));
        pcl_point.b = static_cast<std::uint8_t>(color(2));
        pcl_point_cloud->push_back(pcl_point);
    }

    spdlog::info("Loaded pcl point cloud with {} points", pcl_point_cloud->size());

    spdlog::stopwatch sw;
    auto filtered_point_cloud = ms_slam::slam_core::VoxelGridSamplingPstl<ms_slam::slam_core::PointXYZRGBDescriptor>(point_cloud, 0.05);
    spdlog::info("Filtered point cloud with {} points, time: {:.3f}ms", filtered_point_cloud->size(), sw.elapsed().count() * 1000);

    ms_slam::slam_core::ply_io::SavePointCloud(*filtered_point_cloud, "/home/ubuntu/data/test_filtered.ply", false);

    sw.reset();
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setLeafSize(0.05, 0.05, 0.05);
    voxel_filter.setInputCloud(pcl_point_cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    filtered->reserve(pcl_point_cloud->size());
    voxel_filter.filter(*filtered);
    spdlog::info("Filtered pcl point cloud with {} points , time: {:.3f}ms", filtered->size(), sw.elapsed().count() * 1000);

    auto filtered_pcl_cloud = ms_slam::slam_core::PointCloud<ms_slam::slam_core::PointXYZRGBDescriptor>();
    filtered_pcl_cloud.reserve(filtered->size());
    for (const auto& pcl_point : *filtered) {
        ms_slam::slam_core::PointXYZRGB point(pcl_point.x, pcl_point.y, pcl_point.z, pcl_point.r, pcl_point.g, pcl_point.b);
        filtered_pcl_cloud.push_back(point);
    }
    ms_slam::slam_core::ply_io::SavePointCloud(filtered_pcl_cloud, "/home/ubuntu/data/test_filtered_pcl.ply", false);

    return 0;
}