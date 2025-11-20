#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/bonxai/bonxai.hpp"

namespace ms_slam::slam_core
{

using VoxelBlock = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

/**
 * @brief 基于 Bonxai 的稀疏 VDB 地图封装
 */
class VDBMap {
  public:
    using AccessorType = typename Bonxai::VoxelGrid<VoxelBlock>::Accessor;
    /**
     * @brief 构造 VDB 地图
     * @param voxel_size 体素尺寸
     * @param clipping_distance 地图裁剪距离
     * @param max_points_per_voxel 单个体素最大点数
     * @param neighbor_type 近邻搜索层数，neighbor_type = 1 时在 3x3x3 体素内搜索，2 时在 5x5x5 体素内搜索，依此类推
     * @return 无
     */
    explicit VDBMap(const double voxel_size, const double clipping_distance, const unsigned int max_points_per_voxel, int neighbor_type = 1);

    /**
     * @brief 清空地图内的所有体素
     * @return 无
     */
    void Clear() { map_.clear(Bonxai::ClearOption::CLEAR_MEMORY); }

    /**
     * @brief 判断地图是否为空
     * @return 地图中是否存在有效体素
     */
    [[nodiscard]] bool Empty() const { return map_.activeCellsCount() == 0; }

    /**
     * @brief 将输入点云按位姿变换后加入地图并剔除远点
     * @param points 待加入的点云（传感器系）
     * @param pose 点云对应的位姿（世界系）
     * @return 无
     */
    void Update(const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose);

    /**
     * @brief 直接将点云插入地图
     * @param points 已在世界系中的点云
     * @return 无
     */
    void AddPoints(const std::vector<Eigen::Vector3f>& points);

    /**
     * @brief 按裁剪距离移除远离指定位置的体素
     * @param origin 参考位置（世界系）
     * @return 无
     */
    void RemovePointsFarFromLocation(const Eigen::Vector3d& origin);

    /**
     * @brief 提取当前地图中的全部点云
     * @return 地图内所有点的集合
     */
    [[nodiscard]] std::vector<Eigen::Vector3f> GetPointCloud() const;

    /**
     * @brief 基于体素邻域执行 KNN 搜索
     * @param query 查询点（世界系）
     * @param k 近邻数量
     * @param neighbors 输出的近邻点坐标
     * @param distances 输出的近邻距离
     * @return 成功找到至少一个近邻返回 true，否则返回 false
     *
     * 搜索范围由 neighbor_layers_ 控制，层数为 n 时按 (2n+1)^3 个体素遍历
     */
    bool GetKNearestNeighbors(
        const Eigen::Vector3f& query,
        std::size_t k,
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors,
        std::vector<float>& distances) const;

private:
    double voxel_size_;
    double clipping_distance_;
    unsigned int max_points_per_voxel_;
    int neighbor_layers_;
    double map_resolution_;
    Bonxai::VoxelGrid<VoxelBlock> map_;
    AccessorType accessor_;
    std::vector<Bonxai::CoordT> neighbor_shifts_;
};

}  // namespace ms_slam::slam_core
