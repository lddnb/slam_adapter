#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "slam_core/filter.hpp"
#include "slam_core/utils/ply_io.hpp"
#include "slam_core/point_cloud.hpp"
#include "slam_core/point_types.hpp"

namespace ms_slam::benchmark_detail
{
namespace
{

using ms_slam::slam_core::PointCloudXYZRGB;
using ms_slam::slam_core::PointXYZRGBDescriptor;

constexpr const char* kPlyPath = "/home/ubuntu/data/test_binary.ply";
constexpr int kDefaultOmpThreads = 8;

/**
 * @brief 缓存降采样基准所需的数据集
 */
struct DownsampleBenchmarkDataset
{
    PointCloudXYZRGB::Ptr slam_cloud;
    PointCloudXYZRGB::ConstPtr slam_cloud_const;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pcl_cloud_const;
    int omp_threads{kDefaultOmpThreads};

    /**
     * @brief 构造函数负责读取PLY文件并准备三种算法输入
     * @return 无
     */
    DownsampleBenchmarkDataset()
    {
        spdlog::info("Loading benchmark dataset: {}", kPlyPath);

        slam_cloud = ms_slam::slam_core::ply_io::LoadPointCloud<PointXYZRGBDescriptor>(kPlyPath);
        if (!slam_cloud || slam_cloud->empty()) {
            throw std::runtime_error("Failed to load benchmark point cloud for downsampling.");
        }
        slam_cloud_const = slam_cloud;

        pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl_cloud->reserve(slam_cloud->size());

        // 将 slam_core 点云转换为 PCL 点云，便于直接调用 PCL 滤波器
        for (std::size_t i = 0; i < slam_cloud->size(); ++i) {
            const auto position = slam_cloud->position(i);
            const auto color = slam_cloud->rgb(i);
            pcl::PointXYZRGB pcl_point;
            pcl_point.x = position(0);
            pcl_point.y = position(1);
            pcl_point.z = position(2);
            pcl_point.r = static_cast<std::uint8_t>(color(0));
            pcl_point.g = static_cast<std::uint8_t>(color(1));
            pcl_point.b = static_cast<std::uint8_t>(color(2));
            pcl_cloud->push_back(pcl_point);
        }
        pcl_cloud->width = static_cast<std::uint32_t>(pcl_cloud->size());
        pcl_cloud->height = 1U;
        pcl_cloud->is_dense = false;
        pcl_cloud->sensor_origin_ = Eigen::Vector4f::Zero();
        pcl_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
        pcl_cloud_const = pcl_cloud;
        spdlog::info("pcl point cloud loaded with {} points.", pcl_cloud_const->size());

        spdlog::info("Downsample benchmark dataset loaded with {} points.", slam_cloud->size());
    }
};

/**
 * @brief 访问单例数据集，避免重复加载文件
 * @return 数据集常量引用
 */
const DownsampleBenchmarkDataset& GetDataset()
{
    static const DownsampleBenchmarkDataset dataset{};
    return dataset;
}

/**
 * @brief 将基准参数转换为体素尺寸（单位：米）
 * @param state Google Benchmark 状态对象
 * @return 体素尺寸
 */
double ResolveLeafSize(const benchmark::State& state)
{
    // 使用 state.range(0) 的数值表示毫米，转换为米后便于理解与复现
    return static_cast<double>(state.range(0)) / 1000.0;
}

/**
 * @brief 使用 slam_core OpenMP 实现进行体素滤波基准
 * @param state Google Benchmark 状态对象
 * @return 无
 */
void BM_SlamCore_VoxelGrid_Omp(benchmark::State& state)
{
    const auto& dataset = GetDataset();
    const double leaf_size = ResolveLeafSize(state);

    for (auto _ : state) {
        auto downsampled = ms_slam::slam_core::VoxelGridSamplingOmp<PointXYZRGBDescriptor>(dataset.slam_cloud_const, leaf_size, dataset.omp_threads);
        benchmark::DoNotOptimize(downsampled);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dataset.slam_cloud_const->size()));
}

/**
 * @brief 使用 slam_core PSTL 实现进行体素滤波基准
 * @param state Google Benchmark 状态对象
 * @return 无
 */
void BM_SlamCore_VoxelGrid_Pstl(benchmark::State& state)
{
    const auto& dataset = GetDataset();
    const double leaf_size = ResolveLeafSize(state);

    for (auto _ : state) {
        auto downsampled = ms_slam::slam_core::VoxelGridSamplingPstl<PointXYZRGBDescriptor>(dataset.slam_cloud_const, leaf_size);
        benchmark::DoNotOptimize(downsampled);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dataset.slam_cloud_const->size()));
}

/**
 * @brief 使用 PCL VoxelGrid 实现进行体素滤波基准
 * @param state Google Benchmark 状态对象
 * @return 无
 */
void BM_Pcl_VoxelGrid(benchmark::State& state)
{
    const auto& dataset = GetDataset();
    const double leaf_size = ResolveLeafSize(state);

    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setLeafSize(static_cast<float>(leaf_size), static_cast<float>(leaf_size), static_cast<float>(leaf_size));
    voxel_filter.setInputCloud(dataset.pcl_cloud_const);

    for (auto _ : state) {
        auto filtered = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        filtered->reserve(dataset.pcl_cloud_const->size());
        voxel_filter.filter(*filtered);
        benchmark::DoNotOptimize(filtered->points.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dataset.pcl_cloud_const->size()));
}

}  // namespace
}  // namespace ms_slam::benchmark_detail

// 注册基准任务，使用毫米为单位的参数以提高可读性
BENCHMARK(ms_slam::benchmark_detail::BM_SlamCore_VoxelGrid_Omp)->Arg(100)->Arg(300)->Arg(500)->Unit(benchmark::kMillisecond);
BENCHMARK(ms_slam::benchmark_detail::BM_SlamCore_VoxelGrid_Pstl)->Arg(100)->Arg(300)->Arg(500)->Unit(benchmark::kMillisecond);
BENCHMARK(ms_slam::benchmark_detail::BM_Pcl_VoxelGrid)->Arg(100)->Arg(300)->Arg(500)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
