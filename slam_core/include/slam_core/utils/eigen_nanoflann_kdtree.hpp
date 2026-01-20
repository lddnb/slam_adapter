/**
 * @file eigen_nanoflann_kdtree.hpp
 * @brief 基于 nanoflann 的 Eigen::Vector3f KDTree 轻量封装
 *
 * @note
 * - 本封装面向 3D 点云（Eigen::Vector3f），用于快速 KNN 与半径（RNN）邻域查询。
 * - nanoflann 的 L2 距离度量返回的是“平方距离”，因此本封装的返回距离均为平方距离。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <spdlog/spdlog.h>

#include "slam_core/utils/nanoflann.hpp"

namespace ms_slam::slam_core
{

/**
 * @brief 适配 std::span<const Eigen::Vector3f> 到 nanoflann 数据集接口
 *
 * @note nanoflann 通过该接口在构建/查询时读取点坐标，因此点云内存需在 KDTree 生命周期内保持有效。
 */
struct EigenVector3fCloudView {
    std::span<const Eigen::Vector3f> points;  ///< 点云视图（外部持有内存）

    /**
     * @brief 返回点云点数量
     * @return 点数量
     */
    [[nodiscard]] inline std::size_t kdtree_get_point_count() const noexcept { return points.size(); }

    /**
     * @brief 获取指定点的指定维度坐标
     * @param idx 点索引
     * @param dim 维度索引（0:x, 1:y, 2:z）
     * @return 对应坐标值
     */
    [[nodiscard]] inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const noexcept
    {
        return points[idx](static_cast<Eigen::Index>(dim));
    }

    /**
     * @brief 返回是否提供包围盒（此处不提供，由 nanoflann 自行计算）
     * @tparam BBOX nanoflann 包围盒类型
     * @param bb 包围盒输出（未使用）
     * @return 始终返回 false
     */
    template <class BBOX>
    [[nodiscard]] inline bool kdtree_get_bbox(BBOX& /*bb*/) const noexcept
    {
        return false;
    }
};

/**
 * @brief 面向 Eigen::Vector3f 点云的 3D KDTree 封装（nanoflann 后端）
 *
 * @note
 * - 仅提供“只读点云视图”的 KDTree；如需更新点云，请调用 SetPoints() 后再 Build()。
 * - 所有返回的距离均为平方距离（与 nanoflann::metric_L2 / L2_Simple_Adaptor 一致）。
 *
 * @code
 * 1) 直接使用 std::span<const Eigen::Vector3f>
 * std::vector<Eigen::Vector3f> points = ...;
 * ms_slam::slam_core::EigenNanoFlannKdTree3f tree(std::span<const Eigen::Vector3f>(points.data(), points.size()));
 * Eigen::Vector3f query = ...;
 * std::vector<std::size_t> nn_idx;
 * std::vector<float> nn_dist2;
 * tree.KnnSearch(query, 8, &nn_idx, &nn_dist2);
 *
 * 2) 配合 slam_core::PointCloud（positions_vec3() 已返回 std::span<const Eigen::Vector3f>）
 * auto pts = cloud.positions_vec3();
 * ms_slam::slam_core::EigenNanoFlannKdTree3f tree(pts);
 * @endcode
 */
class EigenNanoFlannKdTree3f
{
  public:
    using Scalar = float;           ///< 标量类型
    using Index = std::size_t;      ///< 点索引类型
    using Point = Eigen::Vector3f;  ///< 点类型

  private:
    using Distance = nanoflann::L2_Simple_Adaptor<Scalar, EigenVector3fCloudView>;
    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<Distance, EigenVector3fCloudView, 3, Index>;

  public:
    /**
     * @brief 构造 KDTree（可选：立即绑定点云并构建索引）
     * @param points 点云视图
     * @param leafMaxSize 叶子节点最大点数（越大构建更快、查询略慢；常用 8~32）
     * @param nThreadBuild 构建线程数（传 0 使用 nanoflann 默认策略）
     * @return 无
     */
    explicit EigenNanoFlannKdTree3f(std::span<const Point> points = {}, std::size_t leafMaxSize = 10, unsigned int nThreadBuild = 0)
    : leafMaxSize_(leafMaxSize),
      nThreadBuild_(nThreadBuild),
      cloud_{points}
    {
        bool res = Build();
        if (!res && !points.empty()) {
            spdlog::error("EigenNanoFlannKdTree3f failed to build KDTree with {} points", points.size());
        }
    }

    /**
     * @brief 绑定新的点云视图（不自动构建索引）
     * @param points 点云视图
     * @return 无
     */
    void SetPoints(std::span<const Point> points) noexcept
    {
        cloud_.points = points;
        // 避免误用旧索引：要求显式 Build() 后才可查询
        index_.reset();
    }

    /**
     * @brief 构建 KDTree 索引
     * @return 是否构建成功（点云为空时返回 false）
     */
    bool Build()
    {
        if (cloud_.points.empty()) {
            index_.reset();
            return false;
        }

        const auto addr = reinterpret_cast<std::uintptr_t>(cloud_.points.data());
        if (addr % alignof(Point) != 0U) {
            // 仅在构建时做一次对齐检查，避免查询路径上的额外开销
            spdlog::error("EigenNanoFlannKdTree3f expects {}-byte alignment, got address {:#x}", alignof(Point), addr);
            index_.reset();
            return false;
        }

        index_ = std::make_unique<KdTree>(
            3,
            cloud_,
            nanoflann::KDTreeSingleIndexAdaptorParams(leafMaxSize_, nanoflann::KDTreeSingleIndexAdaptorFlags::None, nThreadBuild_));
        index_->buildIndex();
        return true;
    }

    /**
     * @brief 返回当前点云点数量
     * @return 点数量
     */
    [[nodiscard]] std::size_t Size() const noexcept { return cloud_.points.size(); }

    /**
     * @brief 根据序号获取源点云中的点坐标
     * @param index 点序号
     * @param outPoint 输出点坐标（非空）
     * @return 是否获取成功
     */
    [[nodiscard]] Point GetPoint(const Index index) const noexcept
    {
        if (index >= cloud_.points.size()) {
            spdlog::error("EigenNanoFlannKdTree3f::GetPoint index out of range: index={}, size={}", index, cloud_.points.size());
            return Point::Zero();
        }
        return cloud_.points[index];
    }

    /**
     * @brief KNN 查询（低分配版本：由调用方提供输出缓存）
     * @param query 查询点
     * @param outIndices 输出：邻居索引数组（长度为 k）
     * @param outSquaredDistances 输出：平方距离数组（长度为 k）
     * @param params nanoflann 查询参数（eps 近似系数等）
     * @return 实际返回邻居数（可能小于 k，当点云数量不足时）
     *
     * @note outIndices 与 outSquaredDistances 长度必须相等且 >0。
     */
    [[nodiscard]] std::size_t KnnSearch(
        const Point& query,
        std::span<Index> outIndices,
        std::span<Scalar> outSquaredDistances,
        const nanoflann::SearchParameters& params = {}) const
    {
        if (!index_) return 0;
        if (outIndices.empty() || outIndices.size() != outSquaredDistances.size()) {
            spdlog::error(
                "EigenNanoFlannKdTree3f::KnnSearch expects non-empty and same-sized outputs, got indices={}, dists={}",
                outIndices.size(),
                outSquaredDistances.size());
            return 0;
        }

        nanoflann::KNNResultSet<Scalar, Index> resultSet(outIndices.size());
        resultSet.init(outIndices.data(), outSquaredDistances.data());
        index_->findNeighbors(resultSet, query.data(), params);
        return static_cast<std::size_t>(resultSet.size());
    }

    /**
     * @brief KNN 查询（便捷版本：内部调整输出 vector 大小）
     * @param query 查询点
     * @param k 近邻数量
     * @param outIndices 输出：邻居索引
     * @param outSquaredDistances 输出：平方距离
     * @param params nanoflann 查询参数
     * @return 实际返回邻居数（可能小于 k）
     */
    [[nodiscard]] std::size_t KnnSearch(
        const Point& query,
        const std::size_t k,
        std::vector<Index>& outIndices,
        std::vector<Scalar>& outSquaredDistances,
        const nanoflann::SearchParameters& params = {}) const
    {
        if (k == 0) {
            outIndices.clear();
            outSquaredDistances.clear();
            return 0;
        }
        outIndices.assign(k, Index{});
        outSquaredDistances.assign(k, Scalar{});
        return KnnSearch(query, std::span<Index>(outIndices), std::span<Scalar>(outSquaredDistances), params);
    }

    /**
     * @brief 半径邻域查询（RNN），以“欧氏半径”输入
     * @param query 查询点
     * @param radius 半径（欧氏距离）
     * @param outIndices 输出：邻居索引（会被清空并写入）
     * @param outSquaredDistances 输出：平方距离（会被清空并写入）
     * @param params nanoflann 查询参数（sorted 控制是否按距离排序）
     * @return 返回邻居数量
     *
     * @note
     * - 内部会将 radius 转为 radius^2 传递给 nanoflann。
     * - 返回的距离为平方距离。
     */
    [[nodiscard]] std::size_t RadiusSearch(
        const Point& query,
        const Scalar radius,
        std::vector<Index>& outIndices,
        std::vector<Scalar>& outSquaredDistances,
        const nanoflann::SearchParameters& params = {}) const
    {
        if (radius < Scalar{0}) {
            spdlog::error("EigenNanoFlannKdTree3f::RadiusSearch expects non-negative radius, got {}", radius);
            return 0;
        }
        if (radius == Scalar{0}) {
            // 允许零半径：只会匹配完全重合的点
            return RadiusSearchSquared(query, Scalar{0}, outIndices, outSquaredDistances, params);
        }
        return RadiusSearchSquared(query, radius * radius, outIndices, outSquaredDistances, params);
    }

    /**
     * @brief 半径邻域查询（RNN），以“平方半径”输入（更高效）
     * @param query 查询点
     * @param squaredRadius 平方半径
     * @param outIndices 输出：邻居索引（会被清空并写入）
     * @param outSquaredDistances 输出：平方距离（会被清空并写入）
     * @param params nanoflann 查询参数
     * @return 返回邻居数量
     */
    [[nodiscard]] std::size_t RadiusSearchSquared(
        const Point& query,
        const Scalar squaredRadius,
        std::vector<Index>& outIndices,
        std::vector<Scalar>& outSquaredDistances,
        const nanoflann::SearchParameters& params = {}) const
    {
        if (!index_) return 0;
        if (squaredRadius < Scalar{0}) {
            spdlog::error("EigenNanoFlannKdTree3f::RadiusSearchSquared expects non-negative squaredRadius, got {}", squaredRadius);
            return 0;
        }

        std::vector<nanoflann::ResultItem<Index, Scalar>> matches;
        matches.reserve(64);  // 经验值：减少小半径查询时的扩容次数

        const std::size_t found = static_cast<std::size_t>(index_->radiusSearch(query.data(), squaredRadius, matches, params));
        outIndices.clear();
        outSquaredDistances.clear();
        outIndices.reserve(found);
        outSquaredDistances.reserve(found);
        for (const auto& item : matches) {
            outIndices.emplace_back(item.first);
            outSquaredDistances.emplace_back(item.second);
        }
        return found;
    }

  private:
    std::size_t leafMaxSize_{10};      ///< KDTree 叶节点最大点数
    unsigned int nThreadBuild_{0};     ///< 构建线程数
    EigenVector3fCloudView cloud_{};   ///< 点云视图适配器
    std::unique_ptr<KdTree> index_{};  ///< nanoflann KDTree 索引
};

}  // namespace ms_slam::slam_core
