#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "slam_core/bonxai/bonxai.hpp"

namespace ms_slam::slam_core
{

using VoxelBlock = std::vector<Eigen::Vector3f>;

struct VDBMap {
    explicit VDBMap(const double voxel_size, const double clipping_distance, const unsigned int max_points_per_voxel);

    void Clear() { map_.clear(Bonxai::ClearOption::CLEAR_MEMORY); }
    bool Empty() const { return map_.activeCellsCount() == 0; }
    void Update(const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose);
    void AddPoints(const std::vector<Eigen::Vector3f>& points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d& origin);
    std::vector<Eigen::Vector3f> Pointcloud() const;
    std::tuple<Eigen::Vector3f, double> GetClosestNeighbor(const Eigen::Vector3f& query) const;

    double voxel_size_;
    double clipping_distance_;
    unsigned int max_points_per_voxel_;
    Bonxai::VoxelGrid<VoxelBlock> map_;

  private:
    using AccessorType = typename Bonxai::VoxelGrid<VoxelBlock>::Accessor;
    AccessorType accessor_;
};

}  // namespace ms_slam::slam_core
