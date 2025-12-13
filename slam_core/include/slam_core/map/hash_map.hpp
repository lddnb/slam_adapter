#pragma once

#include <cassert>

#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <tsl/robin_map.h>

namespace ms_slam::slam_core
{

/**
 * @brief 三维体素索引
 */
struct Voxel {
    Voxel() = default;

    Voxel(short x, short y, short z) : x(x), y(y), z(z) {}

    bool operator==(const Voxel& vox) const { return x == vox.x && y == vox.y && z == vox.z; }

    inline bool operator<(const Voxel& vox) const { return x < vox.x || (x == vox.x && y < vox.y) || (x == vox.x && y == vox.y && z < vox.z); }

    /**
     * @brief 根据点坐标计算体素索引
     * @param point 输入点
     * @param voxel_size 体素尺寸
     * @return 对应的体素索引
     */
    inline static Voxel Coordinates(const Eigen::Vector3f& point, double voxel_size)
    {
        return {static_cast<short>(point.x() / voxel_size), static_cast<short>(point.y() / voxel_size), static_cast<short>(point.z() / voxel_size)};
    }

    short x;
    short y;
    short z;
};

/**
 * @brief 哈希地图体素包含的点集合
 */
struct HashVoxelBlock {
    explicit HashVoxelBlock(int num_points_ = 20) : num_points(num_points_) { points.reserve(num_points_); }

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;

    /**
     * @brief 判断当前体素是否已满
     * @return 是否达到容量上限
     */
    bool IsFull() const { return num_points == points.size(); }

    /**
     * @brief 向体素中添加点
     * @param point 待插入点
     * @return 无
     */
    void AddPoint(const Eigen::Vector3f& point)
    {
        assert(num_points > points.size());
        points.push_back(point);
    }

    /**
     * @brief 查询体素内点的数量
     * @return 当前点数
     */
    inline std::size_t NumPoints() const { return points.size(); }

    /**
     * @brief 查询体素容量
     * @return 容量大小
     */
    inline int Capacity() { return num_points; }

  private:
    std::size_t num_points;
};

using VoxelHashStorage = tsl::robin_map<Voxel, HashVoxelBlock>;

}  // namespace ms_slam::slam_core

namespace std
{

template <>
struct hash<ms_slam::slam_core::Voxel> {
    std::size_t operator()(const ms_slam::slam_core::Voxel& vox) const
    {
        const size_t kP1 = 73856093;
        const size_t kP2 = 19349669;
        const size_t kP3 = 83492791;
        return vox.x * kP1 + vox.y * kP2 + vox.z * kP3;
    }
};
}  // namespace std

namespace ms_slam::slam_core
{
/**
 * @brief 哈希地图配置
 */
struct HashMapConfig {
    double voxel_size = 0.0;                ///< 体素大小
    double map_clipping_distance = 0.0;     ///< 地图裁剪距离
    int voxel_neighborhood = 1;             ///< 搜索邻域半径
    int max_points_per_voxel = 20;          ///< 单个体素最大点数
};

/**
 * @brief 体素哈希地图封装
 */
class VoxelHashMap
{
  private:
    using pair_distance_t = std::tuple<double, Eigen::Vector3f, Voxel>;

    struct Comparator {
        /**
         * @brief 比较距离大小用于优先队列
         * @param left 左侧距离数据
         * @param right 右侧距离数据
         * @return 当左距离更大时返回真
         */
        bool operator()(const pair_distance_t& left, const pair_distance_t& right) const { return std::get<0>(left) < std::get<0>(right); }
    };

    using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

  public:
    /**
     * @brief 构造函数
     * @param config 地图配置
     * @return 无
     */
    explicit VoxelHashMap(const HashMapConfig& config) : config_(config) {
        min_distance_points_ = std::sqrt(config_.voxel_size * config_.voxel_size / config_.max_points_per_voxel);
    }

    /**
     * @brief 插入一个点到体素地图
     * @param point 待插入点（世界系）
     * @return 无
     */
    void AddPoint(const Eigen::Vector3f& point)
    {
        const short kx = static_cast<short>(point[0] / config_.voxel_size);
        const short ky = static_cast<short>(point[1] / config_.voxel_size);
        const short kz = static_cast<short>(point[2] / config_.voxel_size);

        VoxelHashStorage::iterator search = voxel_map_.find(Voxel(kx, ky, kz));

        if (search != voxel_map_.end()) {
            auto& voxel_block = search.value();

            if (!voxel_block.IsFull()) {
                double sq_dist_min_to_points = 10 * config_.voxel_size * config_.voxel_size;
                // 计算该体素中距离当前点最近的平方距离
                for (std::size_t i = 0; i < voxel_block.NumPoints(); ++i) {
                    const auto& voxel_point = voxel_block.points[i];
                    const double sq_dist = (voxel_point - point).squaredNorm();
                    if (sq_dist < sq_dist_min_to_points) {
                        sq_dist_min_to_points = sq_dist;
                    }
                }
                // 距离满足阈值才进行插入
                if (sq_dist_min_to_points > (min_distance_points_ * min_distance_points_)) {
                    voxel_block.AddPoint(point);
                }
            }
        } else {
            HashVoxelBlock block(config_.max_points_per_voxel);
            block.AddPoint(point);
            voxel_map_[Voxel(kx, ky, kz)] = std::move(block);
        }
    }

    /**
     * @brief 搜索近邻体素内的点
     * @param point 查询点
     * @param voxels 可选输出命中的体素索引
     * @return 最近的邻居点集合
     */
    [[nodiscard]] std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> SearchNeighbors(
        const Eigen::Vector3f& point, const int k, std::vector<float>& distances) const
    {
        const short kx = static_cast<short>(point[0] / config_.voxel_size);
        const short ky = static_cast<short>(point[1] / config_.voxel_size);
        const short kz = static_cast<short>(point[2] / config_.voxel_size);

        priority_queue_t priority_queue;

        Voxel voxel_temp(kx, ky, kz);
        for (short kxx = kx - config_.voxel_neighborhood; kxx < kx + config_.voxel_neighborhood + 1; ++kxx) {
            for (short kyy = ky - config_.voxel_neighborhood; kyy < ky + config_.voxel_neighborhood + 1; ++kyy) {
                for (short kzz = kz - config_.voxel_neighborhood; kzz < kz + config_.voxel_neighborhood + 1; ++kzz) {
                    voxel_temp.x = kxx;
                    voxel_temp.y = kyy;
                    voxel_temp.z = kzz;

                    auto search = voxel_map_.find(voxel_temp);
                    if (search != voxel_map_.end()) {
                        const auto& voxel_block = search->second;
                        // 遍历体素内所有点并维护固定容量的大根堆
                        for (std::size_t i = 0; i < voxel_block.NumPoints(); ++i) {
                            const auto& neighbor = voxel_block.points[i];
                            const double distance = (neighbor - point).norm();
                            if (priority_queue.size() == static_cast<size_t>(k)) {
                                if (distance < std::get<0>(priority_queue.top())) {
                                    priority_queue.pop();
                                    priority_queue.emplace(distance, neighbor, voxel_temp);
                                }
                            } else {
                                priority_queue.emplace(distance, neighbor, voxel_temp);
                            }
                        }
                    }
                }
            }
        }

        const auto queue_size = priority_queue.size();
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> closest_neighbors(queue_size);
        distances.resize(queue_size);
        for (size_t i = 0; i < queue_size; ++i) {
            distances[queue_size - 1 - i] = std::get<0>(priority_queue.top());
            closest_neighbors[queue_size - 1 - i] = std::get<1>(priority_queue.top());
            priority_queue.pop();
        }

        return closest_neighbors;
    }

    /**
     * @brief 按裁剪距离清理远处体素
     * @param location 当前参考位置
     * @return 无
     */
    void RemoveDistantVoxels(const Eigen::Vector3d& location)
    {
        if (config_.map_clipping_distance <= 0.0) {
            return;
        }
        const double max_dist_sq = config_.map_clipping_distance * config_.map_clipping_distance;
        std::vector<Voxel> voxels_to_erase;
        voxels_to_erase.reserve(voxel_map_.size());
        for (const auto& pair : voxel_map_) {
            if (pair.second.points.empty()) continue;
            const Eigen::Vector3f& anchor = pair.second.points.front();
            const double sq_dist = (anchor.cast<double>() - location).squaredNorm();
            // 使用体素中第一个点代表该体素中心的近似距离
            if (sq_dist > max_dist_sq) {
                voxels_to_erase.emplace_back(pair.first);
            }
        }
        for (const auto& vox : voxels_to_erase) {
            voxel_map_.erase(vox);
        }
    }

  private:
    VoxelHashStorage voxel_map_;  ///< 内部哈希存储
    HashMapConfig config_;        ///< 配置参数
    double min_distance_points_;  ///< 体素内点间最小距离
};

}  // namespace ms_slam::slam_core
