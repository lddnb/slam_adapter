#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

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
    static std::unique_ptr<VDBMap> Create(const LocalMapParams& params)
    {
        return std::make_unique<VDBMap>(
            params.voxel_size,
            params.map_clipping_distance,
            params.max_points_per_voxel,
            params.voxel_neighborhood);
    }

    static void Knn(VDBMap& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        map.GetKNearestNeighbors(point, k, neighbors, sq_dist);
    }

    static void Update(VDBMap& map, const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose, const Eigen::Vector3d&)
    {
        map.Update(points, pose);
    }
};

template<>
struct MapTraits<VoxelHashMap>
{
    static std::unique_ptr<VoxelHashMap> Create(const LocalMapParams& params)
    {
        HashMapConfig cfg{};
        cfg.voxel_size = params.voxel_size;
        cfg.map_clipping_distance = params.map_clipping_distance;
        cfg.max_points_per_voxel = params.max_points_per_voxel;
        cfg.voxel_neighborhood = params.voxel_neighborhood;
        return std::make_unique<VoxelHashMap>(cfg);
    }

    static void Knn(VoxelHashMap& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        neighbors = map.SearchNeighbors(point, k, sq_dist);
    }

    static void Update(VoxelHashMap& map, const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d&, const Eigen::Vector3d& state_p)
    {
        for (const auto& p : points) {
            map.AddPoint(p);
        }
        map.RemoveDistantVoxels(state_p);
    }
};

template<>
struct MapTraits<thuni::Octree>
{
    static std::unique_ptr<thuni::Octree> Create(const LocalMapParams& params)
    {
        auto map = std::make_unique<thuni::Octree>();
        map->set_min_extent(params.voxel_size / 2);
        map->set_bucket_size(static_cast<size_t>(std::max(params.max_points_per_voxel, 1)));
        map->set_down_size(true);
        return map;
    }

    static void Knn(thuni::Octree& map, const Eigen::Vector3f& point, int k, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors, std::vector<float>& sq_dist)
    {
        map.knnNeighbors(point, k, neighbors, sq_dist);
    }

    static void Update(thuni::Octree& map, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points, const Eigen::Isometry3d&, const Eigen::Vector3d&)
    {
        map.update(points);
    }
};

}  // namespace ms_slam::slam_core
