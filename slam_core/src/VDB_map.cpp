#include "slam_core/VDB_map.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include <spdlog/spdlog.h>
#include "slam_core/bonxai/grid_coord.hpp"

namespace
{
using Bonxai::CoordT;
using NeighborType = ms_slam::slam_core::VDBMap::NeighborType;

constexpr std::array<Bonxai::CoordT, 27> kAllNeighborShifts{
    CoordT{.x = -1, .y = -1, .z = -1}, CoordT{.x = -1, .y = -1, .z = 0}, CoordT{.x = -1, .y = -1, .z = 1},
    CoordT{.x = -1, .y = 0, .z = -1},  CoordT{.x = -1, .y = 0, .z = 0},  CoordT{.x = -1, .y = 0, .z = 1},
    CoordT{.x = -1, .y = 1, .z = -1},  CoordT{.x = -1, .y = 1, .z = 0},  CoordT{.x = -1, .y = 1, .z = 1},

    CoordT{.x = 0, .y = -1, .z = -1},  CoordT{.x = 0, .y = -1, .z = 0},  CoordT{.x = 0, .y = -1, .z = 1},
    CoordT{.x = 0, .y = 0, .z = -1},   CoordT{.x = 0, .y = 0, .z = 0},   CoordT{.x = 0, .y = 0, .z = 1},
    CoordT{.x = 0, .y = 1, .z = -1},   CoordT{.x = 0, .y = 1, .z = 0},   CoordT{.x = 0, .y = 1, .z = 1},

    CoordT{.x = 1, .y = -1, .z = -1},  CoordT{.x = 1, .y = -1, .z = 0},  CoordT{.x = 1, .y = -1, .z = 1},
    CoordT{.x = 1, .y = 0, .z = -1},   CoordT{.x = 1, .y = 0, .z = 0},   CoordT{.x = 1, .y = 0, .z = 1},
    CoordT{.x = 1, .y = 1, .z = -1},   CoordT{.x = 1, .y = 1, .z = 0},   CoordT{.x = 1, .y = 1, .z = 1}};

constexpr uint8_t inner_grid_log2_dim = 2;
constexpr uint8_t leaf_grid_log2_dim = 3;

inline int AbsInt(int value)
{
    return value >= 0 ? value : -value;
}

std::vector<CoordT> BuildNeighborShifts(NeighborType neighbor_type)
{
    std::vector<CoordT> result;
    result.reserve(kAllNeighborShifts.size());

    for (const auto& shift : kAllNeighborShifts) {
        const bool is_center = (shift.x == 0 && shift.y == 0 && shift.z == 0);
        const int non_zero = (AbsInt(shift.x) > 0) + (AbsInt(shift.y) > 0) + (AbsInt(shift.z) > 0);
        bool keep = false;
        switch (neighbor_type) {
            case NeighborType::CENTER:
                keep = is_center;
                break;
            case NeighborType::NEARBY6:
                keep = non_zero <= 1;
                break;
            case NeighborType::NEARBY18:
                keep = non_zero <= 2;
                break;
            case NeighborType::NEARBY26:
                keep = true;
                break;
        }
        if (keep) {
            result.emplace_back(shift);
        }
    }
    return result;
}

}  // namespace

namespace ms_slam::slam_core
{

VDBMap::VDBMap(const double voxel_size, const double clipping_distance, const unsigned int max_points_per_voxel, NeighborType neighbor_type)
: voxel_size_(voxel_size),
  clipping_distance_(clipping_distance),
  max_points_per_voxel_(max_points_per_voxel),
  neighbor_type_(neighbor_type),
  map_(voxel_size, inner_grid_log2_dim, leaf_grid_log2_dim),
  accessor_(map_.createAccessor()),
  neighbor_shifts_(BuildNeighborShifts(neighbor_type))
{
}

bool VDBMap::GetKNearestNeighbors(
    const Eigen::Vector3f& query,
    std::size_t k,
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& neighbors,
    std::vector<float>& distances) const
{
    if (k == 0) {
        spdlog::warn("GetKNearestNeighbors called with k=0, return empty result");
        return false;
    }

    const auto const_accessor = map_.createConstAccessor();
    const Bonxai::CoordT voxel = map_.posToCoord(query);

    std::vector<std::pair<double, Eigen::Vector3f>> candidates;
    candidates.reserve(k * 4);

    // 遍历查询体素及其相邻体素，收集所有候选点
    for (const auto& voxel_shift : neighbor_shifts_) {
        const Bonxai::CoordT query_voxel = voxel + voxel_shift;
        const VoxelBlock* voxel_points = const_accessor.value(query_voxel);
        if (voxel_points == nullptr) {
            continue;
        }
        for (const auto& neighbor : *voxel_points) {
            const double distance = (neighbor - query).norm();
            candidates.emplace_back(distance, neighbor);
        }
    }

    if (candidates.empty()) {
        spdlog::debug("GetKNearestNeighbors locate no points near [{}, {}, {}]", query.x(), query.y(), query.z());
        return false;
    }

    if (candidates.size() > k) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        candidates.resize(k);
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    neighbors.clear();
    neighbors.reserve(candidates.size());
    distances.clear();
    distances.reserve(candidates.size());
    for (const auto& [distance, point] : candidates) {
        neighbors.emplace_back(point);
        distances.emplace_back(distance);
    }
    return true;
}

void VDBMap::AddPoints(const std::vector<Eigen::Vector3f>& points)
{
    const double map_resolution = std::sqrt(voxel_size_ * voxel_size_ / max_points_per_voxel_);
    std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3f& p) {
        const auto voxel_coordinates = map_.posToCoord(p);
        VoxelBlock* voxel_points = accessor_.value(voxel_coordinates, /*create_if_missing=*/true);
        if (voxel_points->size() == max_points_per_voxel_ || std::any_of(voxel_points->cbegin(), voxel_points->cend(), [&](const auto& voxel_point) {
                return (voxel_point - p).norm() < map_resolution;
            })) {
            return;
        }
        voxel_points->reserve(max_points_per_voxel_);
        voxel_points->emplace_back(p);
    });
}

void VDBMap::RemovePointsFarFromLocation(const Eigen::Vector3d& origin)
{
    auto is_too_far_away = [&](const VoxelBlock& block) { return (block.front() - origin.cast<float>()).norm() > clipping_distance_; };

    std::vector<Bonxai::CoordT> keys_to_delete;
    auto& root_map = map_.rootMap();
    for (auto& [key, inner_grid] : root_map) {
        for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
            const int32_t inner_index = *inner_it;
            auto& leaf_grid = inner_grid.cell(inner_index);
            const auto& voxel_block = leaf_grid->cell(leaf_grid->mask().findFirstOn());
            if (is_too_far_away(voxel_block)) {
                inner_grid.mask().setOff(inner_index);
                leaf_grid.reset();
            }
        }
        if (inner_grid.mask().isOff()) {
            keys_to_delete.push_back(key);
        }
    }
    for (const auto& key : keys_to_delete) {
        root_map.erase(key);
    }
}

void VDBMap::Update(const std::vector<Eigen::Vector3f>& points, const Eigen::Isometry3d& pose)
{
    std::vector<Eigen::Vector3f> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(), [&](const auto& point) { return pose.cast<float>() * point; });
    const Eigen::Vector3d& origin = pose.translation();
    AddPoints(points_transformed);
    RemovePointsFarFromLocation(origin);
}

std::vector<Eigen::Vector3f> VDBMap::Pointcloud() const
{
    std::vector<Eigen::Vector3f> point_cloud;
    point_cloud.reserve(map_.activeCellsCount() * max_points_per_voxel_);
    map_.forEachCell(
        [&point_cloud, this](const VoxelBlock& block, const auto&) { point_cloud.insert(point_cloud.end(), block.cbegin(), block.cend()); });
    return point_cloud;
}

}  // namespace ms_slam::slam_core
