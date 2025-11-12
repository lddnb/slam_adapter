#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/bonxai/bonxai.hpp"

namespace ms_slam::slam_core
{

using VoxelBlock = std::vector<Eigen::Vector3f>;

struct VDBMap {
    /**
     * @brief 邻域搜索类型
     */
    enum class NeighborType {
        CENTER,    ///< 仅当前体素
        NEARBY6,   ///< 六联通
        NEARBY18,  ///< 十八联通
        NEARBY26   ///< 二十六联通
    };
    explicit VDBMap(
        const double voxel_size,
        const double clipping_distance,
        const unsigned int max_points_per_voxel,
        NeighborType neighbor_type = NeighborType::NEARBY6);

    void Clear() { map_.clear(Bonxai::ClearOption::CLEAR_MEMORY); }
    bool Empty() const { return map_.activeCellsCount() == 0; }
    void Update(const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose);
    void AddPoints(const std::vector<Eigen::Vector3f>& points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d& origin);
    std::vector<Eigen::Vector3f> Pointcloud() const;
    bool GetKNearestNeighbors(
        const Eigen::Vector3f& query,
        std::size_t k,
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors,
        std::vector<float>& distances) const;

    double voxel_size_;
    double clipping_distance_;
    unsigned int max_points_per_voxel_;
    NeighborType neighbor_type_;
    Bonxai::VoxelGrid<VoxelBlock> map_;

  private:
    using AccessorType = typename Bonxai::VoxelGrid<VoxelBlock>::Accessor;
    AccessorType accessor_;
    std::vector<Bonxai::CoordT> neighbor_shifts_;
};

}  // namespace ms_slam::slam_core
