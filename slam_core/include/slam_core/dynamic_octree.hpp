//
// Created by ubuntu on 2020/6/27.
//

#ifndef LOCALMAPOCTREE_H
#define LOCALMAPOCTREE_H

#include <array>
#include <cmath>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <Eigen/Dense>

#include "slam_core/flann/nanoflann.h"
#include "slam_core/flann/octree.h"

namespace nanoflann::traits
{
template <>
struct access<Eigen::Vector3f, 0>
{
  static float get(const Eigen::Vector3f& p) { return p.x(); }
};

template <>
struct access<Eigen::Vector3f, 1>
{
  static float get(const Eigen::Vector3f& p) { return p.y(); }
};

template <>
struct access<Eigen::Vector3f, 2>
{
  static float get(const Eigen::Vector3f& p) { return p.z(); }
};
}  // namespace nanoflann::traits

namespace
{
using EigenPointCloud = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
using EigenPointCloudPtr = std::shared_ptr<EigenPointCloud>;

struct VoxelKey
{
  int x{0};
  int y{0};
  int z{0};

  bool operator==(const VoxelKey& other) const = default;
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey& key) const noexcept
  {
    constexpr std::size_t kP1 = 73856093;
    constexpr std::size_t kP2 = 19349669;
    constexpr std::size_t kP3 = 83492791;
    return static_cast<std::size_t>(key.x) * kP1 + static_cast<std::size_t>(key.y) * kP2 +
           static_cast<std::size_t>(key.z) * kP3;
  }
};

EigenPointCloudPtr DownsamplePointCloud(const EigenPointCloudPtr& cloud, float voxel_size)
{
  if (!cloud || cloud->empty() || voxel_size <= 0.F)
  {
    return cloud;
  }

  std::unordered_map<VoxelKey, std::pair<Eigen::Vector3f, int>, VoxelKeyHash> accumulator;
  accumulator.reserve(cloud->size());

  for (const auto& point : *cloud)
  {
    VoxelKey key{
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))};

    auto& [sum, count] = accumulator[key];
    if (count == 0)
    {
      sum = point;
      count = 1;
    }
    else
    {
      sum += point;
      ++count;
    }
  }

  auto filtered = std::make_shared<EigenPointCloud>();
  filtered->reserve(accumulator.size());
  for (const auto& [key, value] : accumulator)
  {
    filtered->push_back(value.first / static_cast<float>(value.second));
  }

  return filtered;
}
}  // namespace

struct MapBlock {
    using Point = Eigen::Vector3f;
    using PointCloud = std::vector<Point, Eigen::aligned_allocator<Point>>;
    using Octree = nanoflann::Octree<Point, PointCloud>;
    using PointCloudPtr = std::shared_ptr<PointCloud>;
    using OctreePtr = std::shared_ptr<Octree>;

    MapBlock() = default;
    // 点云数据
    PointCloudPtr pedge_pc_ = nullptr;
    PointCloudPtr psurf_pc_ = nullptr;

    OctreePtr pkdtree_edge_from_block_ = nullptr;
    OctreePtr pkdtree_surf_from_block_ = nullptr;

    // 标志位
    bool bnull_ = true;
    bool bline_null_ = true;
    bool bsurf_null_ = true;
    bool bnewline_points_add_ = false;
    bool bnewsurf_points_add_ = false;

    inline void clear() {
        pedge_pc_.reset();
        psurf_pc_.reset();

        pkdtree_edge_from_block_.reset();
        pkdtree_surf_from_block_.reset();

        bnull_ = true;
        bline_null_ = true;
        bsurf_null_ = true;
        bnewline_points_add_ = false;
        bnewsurf_points_add_ = false;
    }

    inline void insertEdgePoint(const Point &point) {
        if(pedge_pc_ == nullptr) {
            pedge_pc_ = std::make_shared<PointCloud>();
        }
        if(bnull_ or bline_null_) {
            bnull_ = false;
            bline_null_ = false;
        }
        pedge_pc_->push_back(point);
    }

    inline void insertSurfPoint(const Point &point) {
        if(psurf_pc_ == nullptr) {
            psurf_pc_ = std::make_shared<PointCloud>();
        }
        if(bnull_ or bsurf_null_) {
            bnull_ = false;
            bline_null_ = false;
        }
        psurf_pc_->push_back(point);
    }

    inline int edgePointCloudSize() const {
        if(pedge_pc_ == nullptr)
            return 0;
        return static_cast<int>(pedge_pc_->size());
    }

    inline int surfPointCloudSize() const {
        if(psurf_pc_ == nullptr)
            return 0;
        return static_cast<int>(psurf_pc_->size());
    }

    inline bool empty() const { return bnull_; }

    inline bool edgePointsIsEmpty() const { return bline_null_; }

    inline bool surfPointsIsEmpty() const { return bsurf_null_; }

    inline bool haveNewEdgePoints() const { return bnewline_points_add_; }

    inline bool haveNewsurfPoints() const { return bnewsurf_points_add_; }
};

class LocalMap {
public:
    // Usefull types
    using Point = Eigen::Vector3f;
    using PointCloud = std::vector<Point, Eigen::aligned_allocator<Point>>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr const int laserCloudWidth = 21;
    static constexpr const int laserCloudHeight = 21;
    static constexpr int laserCloudDepth = 11;

    static constexpr int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;  // 4851

    static constexpr double voxelResulation = 50;
    static constexpr double halfVoxelResulation = voxelResulation * 0.5;

public:
    LocalMap() {
        // step1: 设置localmap 原点相对于栅格的位置
        origin_ = Eigen::Vector3i(laserCloudWidth * 0.5, laserCloudHeight * 0.5, laserCloudDepth * 0.5);
    }

    Eigen::Vector3i setOrigin(const Eigen::Vector3d &t_w_cur) {
        // 计算当前激光位置相对于栅格地图的位置
        int centerCubeI = int((t_w_cur.x() + halfVoxelResulation) / voxelResulation);
        int centerCubeJ = int((t_w_cur.y() + halfVoxelResulation) / voxelResulation);
        int centerCubeK = int((t_w_cur.z() + halfVoxelResulation) / voxelResulation);

        if(t_w_cur.x() + halfVoxelResulation < 0)
            centerCubeI--;
        if(t_w_cur.y() + halfVoxelResulation < 0)
            centerCubeJ--;
        if(t_w_cur.z() + halfVoxelResulation < 0)
            centerCubeK--;

        origin_.x() = -centerCubeI;
        origin_.y() = -centerCubeJ;
        origin_.z() = -centerCubeK;

        return origin_;
    }  // function setOrigin end

    /// \brief 移动栅格
    /// \param t_w_cur
    /// \return 当前激光点在在栅格中的位置
    Eigen::Vector3i shiftMap(const Eigen::Vector3d &t_w_cur) {

        // 计算当前激光的位置相对于栅格地图的位置
        int centerCubeI = int((t_w_cur.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
        int centerCubeJ = int((t_w_cur.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
        int centerCubeK = int((t_w_cur.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

        if(t_w_cur.x() + halfVoxelResulation < 0)
            centerCubeI--;
        if(t_w_cur.y() + halfVoxelResulation < 0)
            centerCubeJ--;
        if(t_w_cur.z() + halfVoxelResulation < 0)
            centerCubeK--;

        while(centerCubeI < 3) {
            for(int j = 0; j < laserCloudHeight; j++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int i = laserCloudWidth - 1;
                    for(; i >= 1; i--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }

                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeI++;
            origin_.x() = origin_.x() + 1;
        }

        while(centerCubeI >= laserCloudWidth - 3) {
            for(int j = 0; j < laserCloudHeight; j++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int i = 0;
                    for(; i < laserCloudWidth - 1; i++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeI--;
            origin_.x() = origin_.x() - 1;
        }

        while(centerCubeJ < 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int k = 0; k < laserCloudDepth; k++) {

                    int j = laserCloudHeight - 1;

                    for(; j >= 1; j--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
                    }

                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeJ++;
            origin_.y() = origin_.y() + 1;
        }

        while(centerCubeJ >= laserCloudHeight - 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int k = 0; k < laserCloudDepth; k++) {
                    int j = 0;

                    for(; j < laserCloudHeight - 1; j++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeJ--;
            origin_.y() = origin_.y() - 1;
        }

        while(centerCubeK < 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int j = 0; j < laserCloudHeight; j++) {
                    int k = laserCloudDepth - 1;

                    for(; k >= 1; k--) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeK++;
            origin_.z() = origin_.z() + 1;
        }

        while(centerCubeK >= laserCloudDepth - 3) {
            for(int i = 0; i < laserCloudWidth; i++) {
                for(int j = 0; j < laserCloudHeight; j++) {
                    int k = 0;

                    for(; k < laserCloudDepth - 1; k++) {
                        map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                                map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
                    }
                    map_[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k].clear();
                }
            }

            centerCubeK--;
            origin_.z() = origin_.z() - 1;
        }

        return Eigen::Vector3i{centerCubeI, centerCubeJ, centerCubeK};
    }  // function shiftMap

    /// \brief 获取局部5x5栅格局部地图中的line, surf点
    /// \param position　机器人在局部地图中位置
    /// \return std::tuple<int, int> line, surf point size
    std::tuple<int, int> get5x5LocalMapFeatureSize(const Eigen::Vector3i &position) {

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        int laserCloudLineFromMapNum = 0;
        int laserCloudSurfFromMapNum = 0;

        for(int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
            for(int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
                for(int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {

                    if(i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 &&
                       k < laserCloudDepth) {

                        int cubeInd = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        laserCloudLineFromMapNum += map_[cubeInd].edgePointCloudSize();
                        laserCloudSurfFromMapNum += map_[cubeInd].surfPointCloudSize();
                    }
                }
            }
        }

        return std::make_tuple(laserCloudLineFromMapNum, laserCloudSurfFromMapNum);
    }  // function get_localmap_featuresize

    /// \brief 搜索最近邻角点
    /// \param pt_query
    /// \param k_indices
    /// \param k_sqr_distances
    /// \return
    bool nearestKSearchEdgePoint(const Point &pt_query,
                                 std::vector<Point, Eigen::aligned_allocator<Point>> &k_pts,
                                 std::vector<float> &k_sqr_distances) const {

        k_pts.clear();

        int cubeI = int((pt_query.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
        int cubeJ = int((pt_query.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
        int cubeK = int((pt_query.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

        if(pt_query.x() + halfVoxelResulation < 0)
            cubeI--;
        if(pt_query.y() + halfVoxelResulation < 0)
            cubeJ--;
        if(pt_query.z() + halfVoxelResulation < 0)
            cubeK--;

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
        if(map_[cubeInd].pkdtree_edge_from_block_ == nullptr)
            return false;

        const size_t num_results = 5;
        std::vector<size_t> k_indices(num_results);
        k_sqr_distances.resize(num_results);

        map_[cubeInd].pkdtree_edge_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_results, k_indices.data(), k_sqr_distances.data());
        for(auto id : k_indices)
            k_pts.push_back((*(map_[cubeInd].pedge_pc_))[id]);

        return true;
    }  // function nearestKSearchLinePoint

    /**
     * \brief
     * @param [in]  pt_query
     * @param [out] k_pts
     * @param [out] k_sqr_distances
     * @param [in]  num_nearest_search　knn搜寻一定数量的点
     * @param [in]  max_dist_inliner   点到直线的距离的阈值
     * @return
     */
    bool nearestKSearchSpecificEdgePoint(const Point &pt_query,
                                         std::vector<Point, Eigen::aligned_allocator<Point>> &k_pts,
                                         std::vector<float> &k_sqr_distances,
                                         int num_nearest_search,
                                         float max_dist_inliner) const {

        int cubeI = int((pt_query.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
        int cubeJ = int((pt_query.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
        int cubeK = int((pt_query.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

        if(pt_query.x() + halfVoxelResulation < 0)
            cubeI--;
        if(pt_query.y() + halfVoxelResulation < 0)
            cubeJ--;
        if(pt_query.z() + halfVoxelResulation < 0)
            cubeK--;

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
        if(map_[cubeInd].pkdtree_edge_from_block_ == nullptr)
            return false;

        // Get nearest neighbors of the query point
        std::vector<size_t> nearestIndex(num_nearest_search, -1);
        std::vector<float> nearestDist(num_nearest_search, -1.0);
        map_[cubeInd].pkdtree_edge_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_nearest_search, nearestIndex.data(), nearestDist.data());
        // Shortcut to keypoints cloud
        const PointCloud &previousEdgePoints = *(map_[cubeInd].pedge_pc_);

        // to avoid square root when performing comparision
        const float square_max_dist_inliner = max_dist_inliner * max_dist_inliner;

        // take the closest point
        const Point &closest = previousEdgePoints[nearestIndex[0]];
        const Eigen::Vector3f P1 = closest;

        // Loop over neighbors of the neighborhood. For each of them, compute
        // the line between closest point and current point and compute the
        // number of inliers that fit this line.

        std::vector<std::vector<size_t>> inliers_list;

        for(int pt_index = 1; pt_index < num_nearest_search; ++pt_index) {
            // Fit line that links P1 and P2
            const Eigen::Vector3f P2 = previousEdgePoints[nearestIndex[pt_index]];
            Eigen::Vector3f dir = (P2 - P1).normalized();

            // Compute number of inliers of this model
            std::vector<size_t> inlier_index;
            for(int candidate_index = 1; candidate_index < num_nearest_search; ++candidate_index) {
                if(candidate_index == pt_index)
                    inlier_index.push_back(candidate_index);
                else {
                    const Eigen::Vector3f Pcdt = previousEdgePoints[nearestIndex[candidate_index]];
                    if(((Pcdt - P1).cross(dir)).squaredNorm() < square_max_dist_inliner) {
                        inlier_index.push_back(candidate_index);
                    }
                }
            }
            inliers_list.push_back(inlier_index);
        }

        // Keep the line and its inliers with the most inliers
        size_t max_inliers = 0;
        int index_max_inlers = -1;
        for(size_t k = 0; k < inliers_list.size(); ++k) {
            if(inliers_list[k].size() > max_inliers) {
                max_inliers = inliers_list[k].size();
                index_max_inlers = k;
            }
        }

        //        std::vector<Point> &k_pts
        //        std::vector<float> &k_sqr_distances

        // fill
        k_pts.clear();
        k_sqr_distances.clear();

        k_pts.push_back(previousEdgePoints[nearestIndex[0]]);
        k_sqr_distances.push_back(nearestDist[0]);

        for(auto inlier : inliers_list[index_max_inlers]) {
            k_pts.push_back(previousEdgePoints[nearestIndex[inlier]]);
            k_sqr_distances.push_back(nearestDist[inlier]);
        }

        return true;
    }  // function nearestKSearchSpecificLinePoint

    /// \brief 检索最近邻的surf点
    /// \param pt_query
    /// \param k_indices
    /// \param k_sqr_distances
    /// \return
    bool nearestKSearchSurf(const Point &pt_query,
                            std::vector<Point, Eigen::aligned_allocator<Point>> &k_pts,
                            std::vector<float> &k_sqr_distances,
                            int num_nearest_search) const {

        k_pts.clear();

        int cubeI = int((pt_query.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
        int cubeJ = int((pt_query.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
        int cubeK = int((pt_query.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

        if(pt_query.x() + halfVoxelResulation < 0)
            cubeI--;
        if(pt_query.y() + halfVoxelResulation < 0)
            cubeJ--;
        if(pt_query.z() + halfVoxelResulation < 0)
            cubeK--;

        if(!(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
             cubeK < laserCloudDepth)) {
            return false;
        }

        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;

        if(map_[cubeInd].pkdtree_surf_from_block_ == nullptr)
            return false;

        const size_t num_results = num_nearest_search;
        std::vector<size_t> k_indices(num_results);
        k_sqr_distances.resize(num_results);

        map_[cubeInd].pkdtree_surf_from_block_->template knnNeighbors<nanoflann::L2Distance<Point>>(
                pt_query, num_results, k_indices.data(), k_sqr_distances.data());
        for(auto id : k_indices)
            k_pts.push_back((*(map_[cubeInd].psurf_pc_))[id]);
        return true;
    }  // function nearestKSearch_surf

    /// \brief 添加line点, 并进行体素滤波和构建kdtree
    /// \param laserCloudEdgeStack
    void addEdgePointCloud(const PointCloud &laserCloudEdgeStack) {

        // step1: 确定新的一帧点云都分布在哪些block里,并将激光点添加到对应的block中
        std::set<int> blockInd;
        for(const auto &point : laserCloudEdgeStack) {
            int cubeI = int((point.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
            int cubeJ = int((point.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
            int cubeK = int((point.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

            if(point.x() + halfVoxelResulation < 0)
                cubeI--;
            if(point.y() + halfVoxelResulation < 0)
                cubeJ--;
            if(point.z() + halfVoxelResulation < 0)
                cubeK--;

            if(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
               cubeK < laserCloudDepth) {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                blockInd.insert(cubeInd);
                map_[cubeInd].insertEdgePoint(point);
            }
        }

        // TODO: 体素网格过滤和重新构建kdtree
        // step2: 并行检索新添加点分布的block,对block中的点云进行下采样
        //并且重新构造kdtree
        std::vector<int> vblockInd(blockInd.begin(), blockInd.end());

        auto compute_func = [&](const tbb::blocked_range<std::vector<int>::iterator> &range) {
            for(auto &iter : range) {
                map_[iter].pedge_pc_ = DownsamplePointCloud(map_[iter].pedge_pc_, lineRes_);

                if(map_[iter].pkdtree_edge_from_block_ == nullptr)
                    map_[iter].pkdtree_edge_from_block_ = std::make_shared<MapBlock::Octree>();

                map_[iter].pkdtree_edge_from_block_->initialize(*(map_[iter].pedge_pc_));
            }
        };

        tbb::blocked_range<std::vector<int>::iterator> range(vblockInd.begin(), vblockInd.end());
        tbb::parallel_for(range, compute_func);
    }  // function addLinePointCloud

    /// \brief 添加surf点,并进行体素滤波和构建kdtree
    /// \param laserCloudSurfStack
    void addSurfPointCloud(const PointCloud &laserCloudSurfStack) {

        std::set<int> blockInd;
        for(const auto &point : laserCloudSurfStack) {

            int cubeI = int((point.x() + halfVoxelResulation) / voxelResulation) + origin_.x();
            int cubeJ = int((point.y() + halfVoxelResulation) / voxelResulation) + origin_.y();
            int cubeK = int((point.z() + halfVoxelResulation) / voxelResulation) + origin_.z();

            if(point.x() + halfVoxelResulation < 0)
                cubeI--;
            if(point.y() + halfVoxelResulation < 0)
                cubeJ--;
            if(point.z() + halfVoxelResulation < 0)
                cubeK--;

            if(cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
               cubeK < laserCloudDepth) {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;

                blockInd.insert(cubeInd);

                map_[cubeInd].insertSurfPoint(point);
            }
        }

        std::vector<int> vblockInd(blockInd.begin(), blockInd.end());

        auto compute_func = [&](const tbb::blocked_range<std::vector<int>::iterator> &range) {
            for(auto &iter : range) {
                map_[iter].psurf_pc_ = DownsamplePointCloud(map_[iter].psurf_pc_, planeRes_);

                if(map_[iter].pkdtree_surf_from_block_ == nullptr)
                    map_[iter].pkdtree_surf_from_block_ = std::make_shared<MapBlock::Octree>();

                map_[iter].pkdtree_surf_from_block_->initialize(*(map_[iter].psurf_pc_));
            }
        };

        tbb::blocked_range<std::vector<int>::iterator> range(vblockInd.begin(), vblockInd.end());
        tbb::parallel_for(range, compute_func);
    }  // function addSurfPointCloud

    PointCloud getAllLocalMap() const {
        PointCloud laserCloudMap;

        for(const auto &cube : map_) {
            if(cube.pedge_pc_)
                laserCloudMap.insert(laserCloudMap.end(), cube.pedge_pc_->begin(), cube.pedge_pc_->end());
            if(cube.psurf_pc_)
                laserCloudMap.insert(laserCloudMap.end(), cube.psurf_pc_->begin(), cube.psurf_pc_->end());
        }

        return laserCloudMap;
    }  // function get_all_localmap

    PointCloud get5x5LocalMap(const Eigen::Vector3i &position) const {
        PointCloud laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for(int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
            for(int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
                for(int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {

                    if(i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 &&
                       k < laserCloudDepth) {

                        int cubeInd = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        if(map_[cubeInd].pedge_pc_)
                            laserCloudMap.insert(laserCloudMap.end(),
                                                 map_[cubeInd].pedge_pc_->begin(),
                                                 map_[cubeInd].pedge_pc_->end());

                        if(map_[cubeInd].psurf_pc_)
                            laserCloudMap.insert(laserCloudMap.end(),
                                                 map_[cubeInd].psurf_pc_->begin(),
                                                 map_[cubeInd].psurf_pc_->end());
                    }
                }
            }
        }

        return laserCloudMap;
    }  // function get_5x5_localmap

    PointCloud get_5x5_localmap_corner(const Eigen::Vector3i &position) const
    {
        PointCloud laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
        {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
            {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                {

                    if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    {

                        int cubeInd = i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k;
                        if (map_[cubeInd].pedge_pc_)
                            laserCloudMap.insert(laserCloudMap.end(),
                                                 map_[cubeInd].pedge_pc_->begin(),
                                                 map_[cubeInd].pedge_pc_->end());
                    }
                }
            }
        }

        return laserCloudMap;
    } // function get_5x5_localmap_corner

    PointCloud get_5x5_localmap_surf(const Eigen::Vector3i &position) const
    {
        PointCloud laserCloudMap;

        int centerCubeI, centerCubeJ, centerCubeK;
        centerCubeI = position.x();
        centerCubeJ = position.y();
        centerCubeK = position.z();

        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
        {
            for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
            {
                for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
                {

                    if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    {

                        int cubeInd = i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k;

                        if (map_[cubeInd].psurf_pc_)
                            laserCloudMap.insert(laserCloudMap.end(),
                                                 map_[cubeInd].psurf_pc_->begin(),
                                                 map_[cubeInd].psurf_pc_->end());
                    }
                }
            }
        }

        return laserCloudMap;
    } // function get_5x5_localmap_corner

public:
    // localmap
    std::array<MapBlock, laserCloudNum> map_;

    float lineRes_ = 0.2;
    float planeRes_ = 0.4;

    Eigen::Vector3i origin_;
};

#endif  // LOCALMAPOCTREE_H
