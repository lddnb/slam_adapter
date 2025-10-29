#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <Eigen/Core>

#include "slam_core/point_cloud.hpp"

namespace ms_slam::slam_core
{

/**
 * @brief 基于 Eigen::Vector3f 的轻量级八叉树索引
 */
class Octree
{
public:
    using Point = Eigen::Vector3f;
    using PointList = std::vector<Point, Eigen::aligned_allocator<Point>>;

    /**
     * @brief 默认构造函数
     */
    Octree() = default;

    /**
     * @brief 设置叶子节点最大容量
     * @param bucket_size 叶子节点内允许的最大点数
     */
    void SetBucketSize(std::size_t bucket_size)
    {
        bucket_size_ = std::max<std::size_t>(1, bucket_size);
    }

    /**
     * @brief 设置最小体素尺度
     * @param extent 立方体半边长的最小阈值
     */
    void SetMinExtent(float extent)
    {
        min_extent_ = std::max(1e-3f, extent);
    }

    /**
     * @brief 启用或关闭叶子层的降采样策略
     * @param enable 是否启用
     */
    void SetDownsample(bool enable)
    {
        downsample_ = enable;
    }

    /**
     * @brief 清空八叉树
     */
    void Clear()
    {
        root_.reset();
        point_count_ = 0U;
    }

    /**
     * @brief 获取树中点的数量
     * @return 当前缓存的总点数
     */
    [[nodiscard]] std::size_t Size() const
    {
        return point_count_;
    }

    /**
     * @brief 使用点集合重新构建八叉树
     * @param points 输入点云
     */
    void Initialize(std::span<const Point> points)
    {
        Clear();
        if (points.empty())
        {
            spdlog::warn("Octree initialized with empty data set");
            return;
        }

        Point min_bound = Point::Constant(std::numeric_limits<float>::max());
        Point max_bound = Point::Constant(std::numeric_limits<float>::lowest());
        PointList filtered = FilterPoints(points, min_bound, max_bound);

        if (filtered.empty())
        {
            spdlog::warn("Octree discarded all points due to NaN values");
            return;
        }

        const Point half_extent = 0.5F * (max_bound - min_bound);
        const Point centroid = min_bound + half_extent;
        const float extent = std::max(half_extent.maxCoeff(), min_extent_);

        root_ = BuildOctant(centroid, extent, std::move(filtered));
        spdlog::info("Octree initialization complete with {} points", point_count_);
    }

    /**
     * @brief 将新增点插入已有八叉树
     * @param points 输入点云
     */
    void Update(std::span<const Point> points)
    {
        if (points.empty())
        {
            return;
        }
        if (!root_)
        {
            Initialize(points);
            return;
        }

        Point min_bound = Point::Constant(std::numeric_limits<float>::max());
        Point max_bound = Point::Constant(std::numeric_limits<float>::lowest());
        PointList filtered = FilterPoints(points, min_bound, max_bound);

        if (filtered.empty())
        {
            return;
        }

        ExpandRoot(min_bound);
        ExpandRoot(max_bound);
        InsertPoints(*root_, std::move(filtered));
        spdlog::info("Octree update appended {} points, total {}", points.size(), point_count_);
    }

    /**
     * @brief 半径搜索
     * @param query 查询点
     * @param radius 半径
     * @param neighbors 输出近邻点
     * @param squared_distances 输出近邻点与查询点的距离平方
     */
    void RadiusSearch(const Point& query,
                      float radius,
                      PointList& neighbors,
                      std::vector<float>& squared_distances) const
    {
        neighbors.clear();
        squared_distances.clear();

        if (!root_ || radius <= 0.0F)
        {
            return;
        }

        const float radius_sq = radius * radius;
        RadiusSearchRecursive(*root_, query, radius_sq, neighbors, squared_distances);
    }

    /**
     * @brief KNN 搜索
     * @param query 查询点
     * @param k 最近邻数量
     * @param neighbors 输出近邻点
     * @param squared_distances 输出距离平方
     */
    void KnnSearch(const Point& query,
                   std::size_t k,
                   PointList& neighbors,
                   std::vector<float>& squared_distances) const
    {
        neighbors.clear();
        squared_distances.clear();

        if (!root_ || k == 0U)
        {
            return;
        }

        KnnHeap heap(k);
        KnnRecursive(*root_, query, heap);

        neighbors.reserve(heap.Data().size());
        squared_distances.reserve(heap.Data().size());
        for (const auto& entry : heap.Data())
        {
            neighbors.push_back(entry.point);
            squared_distances.push_back(entry.dist_sq);
        }
    }

    /**
     * @brief 将八叉树全部点导出为点云
     * @tparam Descriptor 目标点云的字段描述符
     * @return 智能指针封装的点云拷贝
     */
    template <typename Descriptor>
    [[nodiscard]] typename PointCloud<Descriptor>::Ptr ToPointCloud() const
    {
        using PointCloudT = PointCloud<Descriptor>;
        using Scalar = typename PointCloudT::scalar_type;

        auto cloud = std::make_shared<PointCloudT>();
        if (point_count_ == 0U || !root_)
        {
            spdlog::warn("Octree::ToPointCloud exported empty cloud");
            return cloud;
        }

        PointList buffer;
        buffer.reserve(point_count_);
        CollectPoints(root_.get(), buffer);
        cloud->resize(buffer.size());

        auto positions = cloud->positions();
        constexpr std::size_t dims = PointCloudT::position_dimensions;
        for (std::size_t i = 0; i < buffer.size(); ++i)
        {
            const Point& pt = buffer[i];
            const std::size_t base = i * dims;
            positions[base + 0] = static_cast<Scalar>(pt.x());
            positions[base + 1] = static_cast<Scalar>(pt.y());
            positions[base + 2] = static_cast<Scalar>(pt.z());
        }

        spdlog::info("Octree::ToPointCloud exported {} points", buffer.size());
        return cloud;
    }

private:
    /**
     * @brief 六面体节点，保存当前中心、尺度以及子节点指针
     */
    struct Octant
    {
        Point centroid = Point::Zero();
        float extent = 0.0F;
        PointList points;
        std::array<std::unique_ptr<Octant>, 8> children{};
    };

    /**
     * @brief KNN 堆的单个条目
     */
    struct HeapEntry
    {
        float dist_sq = std::numeric_limits<float>::max();
        Point point = Point::Zero();
    };

    /**
     * @brief 维护固定容量的有序距离堆
     */
    class KnnHeap
    {
    public:
        /**
         * @brief 构造指定容量的有序堆
         * @param capacity 堆容量
         */
        explicit KnnHeap(std::size_t capacity) : capacity_(capacity) {}

        /**
         * @brief 尝试向堆中加入候选点
         * @param point 候选点
         * @param dist_sq 距离平方
         */
        void Add(const Point& point, float dist_sq)
        {
            if (capacity_ == 0)
            {
                return;
            }

            if (entries_.size() == capacity_ && dist_sq >= entries_.back().dist_sq)
            {
                return;
            }

            HeapEntry entry{dist_sq, point};
            const auto it = std::lower_bound(entries_.begin(),
                                             entries_.end(),
                                             entry,
                                             [](const HeapEntry& lhs, const HeapEntry& rhs)
                                             {
                                                 return lhs.dist_sq < rhs.dist_sq;
                                             });
            entries_.insert(it, entry);
            if (entries_.size() > capacity_)
            {
                entries_.pop_back();
            }
        }

        /**
         * @brief 判断堆是否已满
         * @return 是否达到了容量上限
         */
        [[nodiscard]] bool Full() const
        {
            return entries_.size() == capacity_;
        }

        /**
         * @brief 获取当前堆内最劣距离
         * @return 最大距离平方
         */
        [[nodiscard]] float WorstDistance() const
        {
            return entries_.empty() ? std::numeric_limits<float>::max() : entries_.back().dist_sq;
        }

        /**
         * @brief 获取内部存储的数据
         * @return 距离-点对集合
         */
        [[nodiscard]] const std::vector<HeapEntry>& Data() const
        {
            return entries_;
        }

    private:
        std::size_t capacity_;
        std::vector<HeapEntry> entries_;
    };

    /**
     * @brief 判断点是否为有限值
     * @param point 待检测点
     * @return 是否包含 NaN 或 Inf
     */
    [[nodiscard]] static bool IsFinite(const Point& point)
    {
        return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
    }

    /**
     * @brief 计算莫顿编码
     * @param point 输入点
     * @param centroid 分割中心
     * @return 对应八叉体索引
     */
    [[nodiscard]] static std::size_t MortonCode(const Point& point, const Point& centroid)
    {
        std::size_t code = 0U;
        if (point.x() > centroid.x())
        {
            code |= 1U;
        }
        if (point.y() > centroid.y())
        {
            code |= 2U;
        }
        if (point.z() > centroid.z())
        {
            code |= 4U;
        }
        return code;
    }

    /**
     * @brief 过滤非法点并统计包围盒
     * @param points 输入点集合
     * @param min_bound 输出包围盒最小值
     * @param max_bound 输出包围盒最大值
     * @return 合法点集合
     */
    PointList FilterPoints(std::span<const Point> points, Point& min_bound, Point& max_bound) const
    {
        PointList filtered;
        filtered.reserve(points.size());

        bool first_valid = true;
        for (const Point& point : points)
        {
            if (!IsFinite(point))
            {
                continue;
            }

            filtered.push_back(point);
            if (first_valid)
            {
                min_bound = point;
                max_bound = point;
                first_valid = false;
            }
            else
            {
                min_bound = min_bound.cwiseMin(point);
                max_bound = max_bound.cwiseMax(point);
            }
        }

        return filtered;
    }

    /**
     * @brief 构建新八叉体
     * @param centroid 八叉体中心
     * @param extent 半边长
     * @param points 候选点集合
     * @return 构建完成的八叉体
     */
    std::unique_ptr<Octant> BuildOctant(const Point& centroid, float extent, PointList&& points)
    {
        auto octant = std::make_unique<Octant>();
        octant->centroid = centroid;
        octant->extent = extent;

        if (ShouldSplit(points.size(), extent))
        {
            Subdivide(*octant, std::move(points));
        }
        else
        {
            point_count_ += points.size();
            octant->points = std::move(points);
        }
        return octant;
    }

    /**
     * @brief 将点分配到子节点
     * @param octant 当前八叉体
     * @param points 待分配点
     */
    void Subdivide(Octant& octant, PointList&& points)
    {
        static constexpr float kFactor[2] = {-0.5F, 0.5F};
        const float child_extent = 0.5F * octant.extent;

        std::array<PointList, 8> buckets;
        for (const Point& point : points)
        {
            const std::size_t idx = MortonCode(point, octant.centroid);
            buckets[idx].push_back(point);
        }

        for (std::size_t idx = 0U; idx < buckets.size(); ++idx)
        {
            PointList& bucket = buckets[idx];
            if (bucket.empty())
            {
                continue;
            }

            const Point child_centroid(
                octant.centroid.x() + kFactor[(idx & 1U) != 0U] * octant.extent,
                octant.centroid.y() + kFactor[(idx & 2U) != 0U] * octant.extent,
                octant.centroid.z() + kFactor[(idx & 4U) != 0U] * octant.extent);

            octant.children[idx] = BuildOctant(child_centroid, child_extent, std::move(bucket));
        }
    }

    /**
     * @brief 扩展根节点以包围边界点
     * @param boundary 待覆盖点
     */
    void ExpandRoot(const Point& boundary)
    {
        static constexpr float kFactor[2] = {-0.5F, 0.5F};
        while ((boundary - root_->centroid).cwiseAbs().maxCoeff() > root_->extent)
        {
            const float parent_extent = 2.0F * root_->extent;
            const Point parent_centroid(
                root_->centroid.x() + kFactor[(boundary.x() > root_->centroid.x())] * parent_extent,
                root_->centroid.y() + kFactor[(boundary.y() > root_->centroid.y())] * parent_extent,
                root_->centroid.z() + kFactor[(boundary.z() > root_->centroid.z())] * parent_extent);

            auto new_root = std::make_unique<Octant>();
            new_root->centroid = parent_centroid;
            new_root->extent = parent_extent;

            const std::size_t idx = MortonCode(root_->centroid, parent_centroid);
            new_root->children[idx] = std::move(root_);
            root_ = std::move(new_root);
        }
    }

    /**
     * @brief 将点插入指定八叉体
     * @param octant 目标八叉体
     * @param points 待插入点集合
     */
    void InsertPoints(Octant& octant, PointList&& points)
    {
        if (points.empty())
        {
            return;
        }

        if (!HasChildren(octant))
        {
            if (ShouldSplit(octant.points.size() + points.size(), octant.extent))
            {
                point_count_ -= octant.points.size();
                PointList merged = std::move(octant.points);
                merged.insert(merged.end(), points.begin(), points.end());
                Subdivide(octant, std::move(merged));
            }
            else
            {
                if (downsample_ && octant.extent <= 2.0F * min_extent_ &&
                    octant.points.size() > bucket_size_ / 8U)
                {
                    // 叶子节点点数已充足，丢弃新增点降低存储压力
                    return;
                }
                point_count_ += points.size();
                octant.points.insert(octant.points.end(), points.begin(), points.end());
            }
            return;
        }
        static constexpr float kFactor[2] = {-0.5F, 0.5F};
        std::array<PointList, 8> buckets;
        for (const Point& point : points)
        {
            const std::size_t idx = MortonCode(point, octant.centroid);
            buckets[idx].push_back(point);
        }

        const float child_extent = 0.5F * octant.extent;
        for (std::size_t idx = 0U; idx < buckets.size(); ++idx)
        {
            PointList& bucket = buckets[idx];
            if (bucket.empty())
            {
                continue;
            }

            if (!octant.children[idx])
            {
                const Point child_centroid(
                    octant.centroid.x() + kFactor[(idx & 1U) != 0U] * octant.extent,
                    octant.centroid.y() + kFactor[(idx & 2U) != 0U] * octant.extent,
                    octant.centroid.z() + kFactor[(idx & 4U) != 0U] * octant.extent);

                octant.children[idx] = BuildOctant(child_centroid, child_extent, std::move(bucket));
            }
            else
            {
                InsertPoints(*octant.children[idx], std::move(bucket));
            }
        }
    }

    /**
     * @brief 判断八叉体是否拥有子节点
     * @param octant 待检测八叉体
     * @return 是否存在子节点
     */
    [[nodiscard]] bool HasChildren(const Octant& octant) const
    {
        for (const auto& child : octant.children)
        {
            if (child)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief 判断是否需要继续拆分
     * @param point_size 即将存储的点数
     * @param extent 八叉体半边长
     * @return 是否满足拆分条件
     */
    [[nodiscard]] bool ShouldSplit(std::size_t point_size, float extent) const
    {
        return point_size > bucket_size_ && extent > 2.0F * min_extent_;
    }

    /**
     * @brief 递归执行半径搜索
     * @param octant 当前八叉体
     * @param query 查询点
     * @param radius_sq 搜索半径平方
     * @param neighbors 输出邻居点
     * @param squared_distances 输出距离平方
     */
    void RadiusSearchRecursive(const Octant& octant,
                               const Point& query,
                               float radius_sq,
                               PointList& neighbors,
                               std::vector<float>& squared_distances) const
    {
        if (!Overlaps(octant, query, radius_sq))
        {
            return;
        }

        if (!HasChildren(octant))
        {
            for (const Point& point : octant.points)
            {
                const float dist_sq = (point - query).squaredNorm();
                if (dist_sq <= radius_sq)
                {
                    neighbors.push_back(point);
                    squared_distances.push_back(dist_sq);
                }
            }
            return;
        }

        for (const auto& child : octant.children)
        {
            if (child)
            {
                RadiusSearchRecursive(*child, query, radius_sq, neighbors, squared_distances);
            }
        }
    }

    /**
     * @brief 递归执行 KNN 搜索
     * @param octant 当前八叉体
     * @param query 查询点
     * @param heap 最近邻堆
     */
    void KnnRecursive(const Octant& octant, const Point& query, KnnHeap& heap) const
    {
        if (heap.Full() && !Overlaps(octant, query, heap.WorstDistance()))
        {
            return;
        }

        if (!HasChildren(octant))
        {
            for (const Point& point : octant.points)
            {
                const float dist_sq = (point - query).squaredNorm();
                heap.Add(point, dist_sq);
            }
            return;
        }

        const std::size_t primary_index = MortonCode(query, octant.centroid);
        static constexpr std::array<std::array<std::size_t, 7>, 8> ordered_indices{
            std::array<std::size_t, 7>{1, 2, 4, 3, 5, 6, 7},
            std::array<std::size_t, 7>{0, 3, 5, 2, 4, 7, 6},
            std::array<std::size_t, 7>{0, 3, 6, 1, 4, 7, 5},
            std::array<std::size_t, 7>{1, 2, 7, 0, 5, 6, 4},
            std::array<std::size_t, 7>{0, 5, 6, 1, 2, 7, 3},
            std::array<std::size_t, 7>{1, 4, 7, 0, 3, 6, 2},
            std::array<std::size_t, 7>{2, 4, 7, 0, 3, 5, 1},
            std::array<std::size_t, 7>{3, 5, 6, 1, 2, 4, 0}};

        if (octant.children[primary_index])
        {
            KnnRecursive(*octant.children[primary_index], query, heap);
        }

        for (const std::size_t offset : ordered_indices[primary_index])
        {
            const auto& child = octant.children[offset];
            if (!child)
            {
                continue;
            }
            if (heap.Full() && !Overlaps(*child, query, heap.WorstDistance()))
            {
                continue;
            }
            KnnRecursive(*child, query, heap);
        }
    }

    /**
     * @brief 判断八叉体与球是否相交
     * @param octant 当前八叉体
     * @param query 球心
     * @param radius_sq 半径平方
     * @return 是否相交
     */
    [[nodiscard]] bool Overlaps(const Octant& octant, const Point& query, float radius_sq) const
    {
        const Point delta = (query - octant.centroid).cwiseAbs() - Point::Constant(octant.extent);
        Point clamped = delta.cwiseMax(Point::Zero());
        return clamped.squaredNorm() <= radius_sq;
    }

private:
    /**
     * @brief 递归收集八叉树所有叶节点的点
     * @param octant 当前八叉体指针
     * @param output 点集合
     */
    void CollectPoints(const Octant* octant, PointList& output) const
    {
        if (octant == nullptr)
        {
            return;
        }
        if (!HasChildren(*octant))
        {
            output.insert(output.end(), octant->points.begin(), octant->points.end());
            return;
        }
        for (const auto& child : octant->children)
        {
            if (child)
            {
                CollectPoints(child.get(), output);
            }
        }
    }

    std::unique_ptr<Octant> root_;
    std::size_t point_count_ = 0U;
    std::size_t bucket_size_ = 32U;
    float min_extent_ = 0.2F;
    bool downsample_ = true;
};

}  // namespace ms_slam::slam_core
