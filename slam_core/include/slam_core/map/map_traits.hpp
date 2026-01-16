#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/config.hpp"
#include "slam_core/map/VDB_map.hpp"
#include "slam_core/map/Octree.hpp"
#include "slam_core/map/hash_map.hpp"

namespace ms_slam::slam_core
{
template<typename MapT>
struct MapTraits;

template<>
struct MapTraits<VDBMap>
{
    /**
     * @brief 创建 VDBMap 地图实例
     * 
     * @param params 地图参数
     * @return std::unique_ptr<VDBMap> 
     */
    static std::unique_ptr<VDBMap> Create(const LocalMapParams& params)
    {
        return std::make_unique<VDBMap>(
            params.voxel_size,
            params.map_clipping_distance,
            params.max_points_per_voxel,
            params.voxel_neighborhood);
    }

    /**
     * @brief 查询 K 近邻点（世界系），并返回平方距离以统一不同地图的距离语义
     * @param map 地图实例
     * @param point 查询点（世界系）
     * @param k 近邻数量
     * @param neighbors 输出近邻点（世界系）
     * @param sq_dist 输出平方距离（单位：m^2）
     * @return 无
     */
    static void Knn(VDBMap& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        map.GetKNearestNeighbors(point, k, neighbors, sq_dist);
    }

    /**
     * @brief 增量更新地图（输入点为雷达系，内部会变换到世界系写入）
     * @param map 地图实例
     * @param points 输入点（雷达系）
     * @param pose 世界到雷达的位姿（世界系）
     * @param unused 未使用的平移参数
     * @return 无
     */
    static void Update(VDBMap& map, const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose, const Eigen::Vector3d&)
    {
        map.Update(points, pose);
    }

    static std::vector<Eigen::Vector3f> GetPointCloud(const VDBMap& map)
    {
        return map.GetPointCloud();
    }
};

template<>
struct MapTraits<VoxelHashMap>
{
    /**
     * @brief 创建 VoxelHashMap 地图实例
     * 
     * @param params 地图参数
     * @return std::unique_ptr<VoxelHashMap> 
     */
    static std::unique_ptr<VoxelHashMap> Create(const LocalMapParams& params)
    {
        HashMapConfig cfg{};
        cfg.voxel_size = params.voxel_size;
        cfg.map_clipping_distance = params.map_clipping_distance;
        cfg.max_points_per_voxel = params.max_points_per_voxel;
        cfg.voxel_neighborhood = params.voxel_neighborhood;
        return std::make_unique<VoxelHashMap>(cfg);
    }

    /**
     * @brief 查询 K 近邻点（世界系），并返回平方距离以统一不同地图的距离语义
     * @param map 地图实例
     * @param point 查询点（世界系）
     * @param k 近邻数量
     * @param neighbors 输出近邻点（世界系）
     * @param sq_dist 输出平方距离（单位：m^2）
     * @return 无
     */
    static void Knn(VoxelHashMap& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        neighbors = map.SearchNeighbors(point, k, sq_dist);
    }

    /**
     * @brief 增量更新地图（将雷达系点变换到世界系写入），并按裁剪距离清理远处体素
     * @param map 地图实例
     * @param points 输入点（雷达系）
     * @param pose 世界到雷达的位姿（世界系）
     * @param state_p 当前位姿平移（世界系），用于裁剪
     * @return 无
     */
    static void Update(VoxelHashMap& map, const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose, const Eigen::Vector3d& state_p)
    {
        const Eigen::Isometry3f world_T_lidar = pose.cast<float>();
        for (const auto& p : points) {
            map.AddPoint(world_T_lidar * p);
        }
        map.RemoveDistantVoxels(state_p);
    }
};

template<>
struct MapTraits<thuni::Octree>
{
    /**
     * @brief 创建 Octree 地图实例
     * 
     * @param params 地图参数
     * @return std::unique_ptr<thuni::Octree> 
     */
    static std::unique_ptr<thuni::Octree> Create(const LocalMapParams& params)
    {
        auto map = std::make_unique<thuni::Octree>();
        map->set_min_extent(params.voxel_size / 2);
        map->set_bucket_size(static_cast<size_t>(std::max(params.max_points_per_voxel, 1)));
        map->set_down_size(true);
        return map;
    }

    /**
     * @brief 查询 K 近邻点（世界系），距离语义为平方距离（单位：m^2）
     * @param map 地图实例
     * @param point 查询点（世界系）
     * @param k 近邻数量
     * @param neighbors 输出近邻点（世界系）
     * @param sq_dist 输出平方距离（单位：m^2）
     * @return 无
     */
    static void Knn(thuni::Octree& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        map.knnNeighbors(point, k, neighbors, sq_dist);
    }

    /**
     * @brief 增量更新地图（将雷达系点变换到世界系写入）
     * @param map 地图实例
     * @param points 输入点（雷达系）
     * @param pose 世界到雷达的位姿（世界系）
     * @param unused 未使用的平移参数
     * @return 无
     */
    static void Update(thuni::Octree& map, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points, const Eigen::Isometry3d& pose, const Eigen::Vector3d&)
    {
        const Eigen::Isometry3f world_T_lidar = pose.cast<float>();
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_world;
        points_world.reserve(points.size());
        for (const auto& p : points) {
            points_world.emplace_back(world_T_lidar * p);
        }
        map.update(points_world);
    }
};

}  // namespace ms_slam::slam_core
