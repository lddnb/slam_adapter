#pragma once

#include <iostream>
#include <math.h>
#include <thread>
#include <fstream>
#include <vector>
#include <queue>

#include <Eigen/Core>
#include <tsl/robin_map.h>
#include "slam_core/point_cloud.hpp"

namespace ms_slam::slam_core
{

struct voxel {
    voxel() = default;

    voxel(short x, short y, short z) : x(x), y(y), z(z) {}

    bool operator==(const voxel& vox) const { return x == vox.x && y == vox.y && z == vox.z; }

    inline bool operator<(const voxel& vox) const { return x < vox.x || (x == vox.x && y < vox.y) || (x == vox.x && y == vox.y && z < vox.z); }

    inline static voxel coordinates(const Eigen::Vector3f& point, double voxel_size)
    {
        return {short(point.x() / voxel_size), short(point.y() / voxel_size), short(point.z() / voxel_size)};
    }

    short x;
    short y;
    short z;
};

struct voxelBlock {
    explicit voxelBlock(int num_points_ = 20) : num_points(num_points_) { points.reserve(num_points_); }

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;

    bool IsFull() const { return num_points == points.size(); }

    void AddPoint(const Eigen::Vector3f& point)
    {
        assert(num_points > points.size());
        points.push_back(point);
    }

    inline int NumPoints() const { return points.size(); }

    inline int Capacity() { return num_points; }

  private:
    int num_points;
};

typedef tsl::robin_map<voxel, voxelBlock> voxelHashMap;

}  // namespace ms_slam::slam_core

namespace std
{

template <>
struct hash<ms_slam::slam_core::voxel> {
    std::size_t operator()(const ms_slam::slam_core::voxel& vox) const
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
using pair_distance_t = std::tuple<double, Eigen::Vector3f, voxel>;

struct comparator {
    bool operator()(const pair_distance_t& left, const pair_distance_t& right) const { return std::get<0>(left) < std::get<0>(right); }
};

using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, comparator>;

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> searchNeighbors(
    const voxelHashMap& map,
    const Eigen::Vector3f& point,
    int nb_voxels_visited,
    double size_voxel_map,
    int max_num_neighbors,
    int threshold_voxel_capacity,
    std::vector<voxel>* voxels)
{
    if (voxels != nullptr) voxels->reserve(max_num_neighbors);

    short kx = static_cast<short>(point[0] / size_voxel_map);
    short ky = static_cast<short>(point[1] / size_voxel_map);
    short kz = static_cast<short>(point[2] / size_voxel_map);

    priority_queue_t priority_queue;

    voxel voxel_temp(kx, ky, kz);
    for (short kxx = kx - nb_voxels_visited; kxx < kx + nb_voxels_visited + 1; ++kxx) {
        for (short kyy = ky - nb_voxels_visited; kyy < ky + nb_voxels_visited + 1; ++kyy) {
            for (short kzz = kz - nb_voxels_visited; kzz < kz + nb_voxels_visited + 1; ++kzz) {
                voxel_temp.x = kxx;
                voxel_temp.y = kyy;
                voxel_temp.z = kzz;

                auto search = map.find(voxel_temp);
                if (search != map.end()) {
                    const auto& voxel_block = search.value();
                    if (voxel_block.NumPoints() < threshold_voxel_capacity) continue;
                    for (int i = 0; i < voxel_block.NumPoints(); ++i) {
                        auto& neighbor = voxel_block.points[i];
                        double distance = (neighbor - point).norm();
                        if (priority_queue.size() == max_num_neighbors) {
                            if (distance < std::get<0>(priority_queue.top())) {
                                priority_queue.pop();
                                priority_queue.emplace(distance, neighbor, voxel_temp);
                            }
                        } else
                            priority_queue.emplace(distance, neighbor, voxel_temp);
                    }
                }
            }
        }
    }

    auto size = priority_queue.size();
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> closest_neighbors(size);
    if (voxels != nullptr) {
        voxels->resize(size);
    }
    for (auto i = 0; i < size; ++i) {
        closest_neighbors[size - 1 - i] = std::get<1>(priority_queue.top());
        if (voxels != nullptr) (*voxels)[size - 1 - i] = std::get<2>(priority_queue.top());
        priority_queue.pop();
    }

    return closest_neighbors;
}

void addPointToMap(
    voxelHashMap& map,
    const Eigen::Vector3f& point,
    double voxel_size,
    int max_num_points_in_voxel,
    double min_distance_points,
    int min_num_points)
{
    short kx = static_cast<short>(point[0] / voxel_size);
    short ky = static_cast<short>(point[1] / voxel_size);
    short kz = static_cast<short>(point[2] / voxel_size);

    voxelHashMap::iterator search = map.find(voxel(kx, ky, kz));

    if (search != map.end()) {
        auto& voxel_block = (search.value());

        if (!voxel_block.IsFull()) {
            double sq_dist_min_to_points = 10 * voxel_size * voxel_size;
            for (int i = 0; i < voxel_block.NumPoints(); ++i) {
                auto& _point = voxel_block.points[i];
                const double sq_dist = (_point - point).squaredNorm();
                if (sq_dist < sq_dist_min_to_points) {
                    sq_dist_min_to_points = sq_dist;
                }
            }
            if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
                    voxel_block.AddPoint(point);
                    // addPointToPcl(points_world, point, intensity, p_frame);
                }
            }
        }
    } else {
        if (min_num_points <= 0) {
            voxelBlock block(max_num_points_in_voxel);
            block.AddPoint(point);
            map[voxel(kx, ky, kz)] = std::move(block);
        }
    }
}

}  // namespace ms_slam::slam_core
