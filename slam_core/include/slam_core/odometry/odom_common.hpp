#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "slam_core/sensor/image.hpp"
#include "slam_core/sensor/imu.hpp"
#include "slam_core/sensor/point_cloud.hpp"

namespace ms_slam::slam_core
{
using PointType = PointXYZITDescriptor;
using PointCloudType = PointCloud<PointType>;

struct SyncData {
    PointCloudType::ConstPtr lidar_data;
    double lidar_beg_time;
    double lidar_end_time;
    Image image_data;
    std::vector<IMU> imu_data;
};

/**
 * @brief 按时间查询位姿的回调
 */
using PoseAtTimeFn = std::function<std::optional<Eigen::Isometry3d>(double timestamp)>;

struct Match {
    Eigen::Vector3d p;
    Eigen::Vector4d n;
    double dist2plane;
    double confidence = 1.0;

    Match() = default;
    Match(const Eigen::Vector3d& p_, const Eigen::Vector4d& n_, double dist) : p(p_), n(n_), dist2plane(dist) {}

    inline static double Dist2Plane(const Eigen::Vector4d& normal, const Eigen::Vector3d& point) { return normal.head<3>().dot(point) + normal(3); }
};

using Matches = std::vector<Match>;

inline bool EstimatePlane(Eigen::Vector4d& pabcd, const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& pts, const double& thresh)
{
    const int N = static_cast<int>(pts.size());
    if (N < 3) return false;

    Eigen::Matrix<double, Eigen::Dynamic, 3> neighbors(N, 3);
    for (int i = 0; i < N; ++i) {
        neighbors.row(i) = pts[static_cast<std::size_t>(i)].cast<double>();
    }

    const Eigen::Vector3d centroid = neighbors.colwise().mean();
    neighbors.rowwise() -= centroid.transpose();

    const Eigen::Matrix3d cov = (neighbors.transpose() * neighbors) / static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
    if (eigensolver.info() != Eigen::Success) return false;

    const Eigen::Vector3d normal = eigensolver.eigenvectors().col(0);
    const double d = -normal.dot(centroid);

    pabcd.head<3>() = normal;
    pabcd(3) = d;

    for (const auto& p : pts) {
        const double distance = normal.dot(p.cast<double>()) + d;
        if (std::abs(distance) > thresh) return false;
    }

    return true;
}

inline std::span<Eigen::Vector3f> MakeVec3Span(std::span<float> data)
{
    if (data.size() % 3 != 0) {
        spdlog::error("MakeVec3Span expects data.size() % 3 == 0, got {}", data.size());
        return {};
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(data.data());
    if (addr % alignof(Eigen::Vector3f) != 0U) {
        spdlog::error("MakeVec3Span expects {}-byte alignment, got address {:#x}", alignof(Eigen::Vector3f), addr);
        return {};
    }
    return {reinterpret_cast<Eigen::Vector3f*>(data.data()), data.size() / 3};
}

/**
 * @brief 点云滤波器参数集合
 */
struct LidarFilterOptions
{
    bool distance_active{false};
    double min_distance{0.0};
    bool rate_active{false};
    std::size_t sampling_stride{1};
    bool fov_active{false};
    double half_angle_rad{3.141592653589793};
};

inline Eigen::Array3i FastFloor(const Eigen::Array3f& pt)
{
    const Eigen::Array3i ncoord = pt.cast<int>();
    return ncoord - (pt < ncoord.cast<float>()).cast<int>();
}

template<typename RandomAccessIterator, typename Compare>
void QuickSortOmpImpl(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp)
{
    const std::ptrdiff_t n = std::distance(first, last);
    if (n < 1024) {
        std::sort(first, last, comp);
        return;
    }

    const auto median3 = [&](const auto& a, const auto& b, const auto& c, const Compare& cmp) {
        return cmp(a, b) ? (cmp(b, c) ? b : (cmp(a, c) ? c : a)) : (cmp(a, c) ? a : (cmp(b, c) ? c : b));
    };

    const int offset = static_cast<int>(n / 8);
    const auto m1 = median3(*first, *(first + offset), *(first + offset * 2), comp);
    const auto m2 = median3(*(first + offset * 3), *(first + offset * 4), *(first + offset * 5), comp);
    const auto m3 = median3(*(first + offset * 6), *(first + offset * 7), *(last - 1), comp);

    auto pivot = median3(m1, m2, m3, comp);
    auto middle1 = std::partition(first, last, [&](const auto& val) { return comp(val, pivot); });
    auto middle2 = std::partition(middle1, last, [&](const auto& val) { return !comp(pivot, val); });

#ifndef _MSC_VER
#pragma omp task
    QuickSortOmpImpl(first, middle1, comp);

#pragma omp task
    QuickSortOmpImpl(middle2, last, comp);
#else
    std::sort(first, middle1, comp);
    std::sort(middle2, last, comp);
#endif
}

template<typename RandomAccessIterator, typename Compare>
void QuickSortOmp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads)
{
#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp single nowait
        {
            QuickSortOmpImpl(first, last, comp);
        }
    }
#else
    std::sort(first, last, comp);
#endif
}

namespace detail
{
template<typename Descriptor>
struct VoxelAccumulator
{
    Eigen::Array3d position_sum;
    Eigen::Array3d normal_sum;
    Eigen::Array3d rgb_sum;
    double intensity_sum;
    double timestamp_sum;
    double curvature_sum;
    std::size_t count;

    VoxelAccumulator() { reset(); }

    void reset()
    {
        position_sum.setZero();
        normal_sum.setZero();
        rgb_sum.setZero();
        intensity_sum = 0.0;
        timestamp_sum = 0.0;
        curvature_sum = 0.0;
        count = 0;
    }

    template<typename Cloud>
    void accumulate(const Cloud& cloud, std::size_t index)
    {
        const auto pos = cloud.position(index);
        position_sum += pos.template cast<double>().array();
        ++count;

        if constexpr (has_field_v<IntensityTag, Descriptor>) {
            intensity_sum += static_cast<double>(cloud.intensity(index));
        }
        if constexpr (has_field_v<TimestampTag, Descriptor>) {
            timestamp_sum += static_cast<double>(cloud.timestamp(index));
        }
        if constexpr (has_field_v<CurvatureTag, Descriptor>) {
            curvature_sum += static_cast<double>(cloud.curvature(index));
        }
        if constexpr (has_field_v<NormalTag, Descriptor>) {
            const auto normal = cloud.normal(index);
            normal_sum += normal.template cast<double>().array();
        }
        if constexpr (has_field_v<RGBTag, Descriptor>) {
            const auto rgb = cloud.rgb(index);
            static constexpr std::size_t dims = field_descriptor_t<RGBTag, Descriptor>::dimensions;
            for (std::size_t i = 0; i < dims; ++i) {
                rgb_sum(static_cast<Eigen::Index>(i)) += static_cast<double>(rgb(static_cast<Eigen::Index>(i)));
            }
        }
    }
};

template<typename Descriptor>
void write_voxel_average(PointCloud<Descriptor>& cloud, std::size_t index, const VoxelAccumulator<Descriptor>& accum)
{
    if (accum.count == 0) {
        return;
    }

    using CloudType = PointCloud<Descriptor>;
    using Scalar = typename CloudType::scalar_type;
    const double inv = 1.0 / static_cast<double>(accum.count);

    cloud.position(index) = (accum.position_sum * inv).template cast<Scalar>();

    if constexpr (has_field_v<IntensityTag, Descriptor>) {
        using IntensityScalar = typename CloudType::template FieldScalarT<IntensityTag>;
        cloud.intensity(index) = static_cast<IntensityScalar>(accum.intensity_sum * inv);
    }
    if constexpr (has_field_v<TimestampTag, Descriptor>) {
        using TimestampScalar = typename CloudType::template FieldScalarT<TimestampTag>;
        cloud.timestamp(index) = static_cast<TimestampScalar>(accum.timestamp_sum * inv);
    }
    if constexpr (has_field_v<CurvatureTag, Descriptor>) {
        using CurvatureScalar = typename CloudType::template FieldScalarT<CurvatureTag>;
        cloud.curvature(index) = static_cast<CurvatureScalar>(accum.curvature_sum * inv);
    }
    if constexpr (has_field_v<NormalTag, Descriptor>) {
        auto normal_out = cloud.normal(index);
        constexpr std::size_t dims = field_descriptor_t<NormalTag, Descriptor>::dimensions;
        for (std::size_t i = 0; i < dims; ++i) {
            normal_out(static_cast<Eigen::Index>(i)) =
                static_cast<typename CloudType::template FieldScalarT<NormalTag>>(accum.normal_sum(static_cast<Eigen::Index>(i)) * inv);
        }
    }
    if constexpr (has_field_v<RGBTag, Descriptor>) {
        auto rgb_out = cloud.rgb(index);
        constexpr std::size_t dims = field_descriptor_t<RGBTag, Descriptor>::dimensions;
        for (std::size_t i = 0; i < dims; ++i) {
            const double value = accum.rgb_sum(static_cast<Eigen::Index>(i)) * inv;
            const long rounded = std::lround(value);
            const long clamped = std::clamp<long>(rounded, 0L, 255L);
            rgb_out(static_cast<Eigen::Index>(i)) = static_cast<std::uint8_t>(clamped);
        }
    }
}
}  // namespace detail

template<typename Descriptor>
typename PointCloud<Descriptor>::Ptr ApplyLidarFilters(const typename PointCloud<Descriptor>::ConstPtr& cloud, const LidarFilterOptions& options)
{
    using Cloud = PointCloud<Descriptor>;
    if (!cloud) {
        spdlog::error("ApplyLidarFilters received null cloud pointer.");
        return std::make_shared<Cloud>();
    }
    if (cloud->empty()) {
        return std::make_shared<Cloud>();
    }

    const bool distance_enabled = options.distance_active && options.min_distance > 0.0;
    const float min_distance_sq = distance_enabled ? static_cast<float>(options.min_distance * options.min_distance) : 0.0F;

    const std::size_t stride = options.sampling_stride == 0 ? 1 : options.sampling_stride;
    const bool rate_enabled = options.rate_active && stride > 1;

    const bool fov_enabled = options.fov_active && options.half_angle_rad > 0.0;
    const float half_angle = static_cast<float>(options.half_angle_rad);

    const std::size_t point_count = cloud->size();
    std::vector<std::size_t> indices(point_count);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<std::size_t> kept_indices(point_count);
    std::atomic_size_t kept_count{0};

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](std::size_t idx) {
        bool pass = true;
        const auto position = cloud->position(idx);
        Eigen::Vector3f lidar_point = position.template cast<float>();

        if (distance_enabled) {
            const float norm_sq = lidar_point.squaredNorm();
            if (norm_sq <= min_distance_sq) {
                pass = false;
            }
        }
        if (pass && rate_enabled) {
            if (idx % stride != 0) {
                pass = false;
            }
        }
        if (pass && fov_enabled) {
            const float azimuth = std::atan2(lidar_point.y(), lidar_point.x());
            if (std::fabs(azimuth) >= half_angle) {
                pass = false;
            }
        }
        if (pass) {
            const std::size_t slot = kept_count.fetch_add(1, std::memory_order_relaxed);
            kept_indices[slot] = idx;
        }
    });

    const std::size_t valid_count = kept_count.load(std::memory_order_relaxed);
    if (valid_count == 0) {
        return std::make_shared<Cloud>();
    }
    kept_indices.resize(valid_count);

    auto filtered = std::make_shared<Cloud>(cloud->extract(kept_indices));
    return filtered;
}

template<typename Descriptor>
typename PointCloud<Descriptor>::Ptr VoxelGridSamplingPstl(const typename PointCloud<Descriptor>::ConstPtr& points, double leaf_size)
{
    using Cloud = PointCloud<Descriptor>;
    if (!points || points->empty()) {
        return std::make_shared<Cloud>();
    }

    const double inv_leaf_size = 1.0 / leaf_size;

    constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
    constexpr int coord_bit_size = 21;
    constexpr std::size_t coord_bit_mask = (1 << 21) - 1;
    constexpr int coord_offset = 1 << (coord_bit_size - 1);

    const std::size_t point_count = points->size();
    std::vector<std::pair<std::uint64_t, std::size_t>> coord_pt(point_count);

    std::for_each(std::execution::par, coord_pt.begin(), coord_pt.end(), [&](auto& pair) {
        const std::size_t i = static_cast<std::size_t>(&pair - coord_pt.data());
        const auto pos = points->position(i);
        const Eigen::Array3f pt_f = pos.array();
        const Eigen::Array3i coord = FastFloor(pt_f * static_cast<float>(inv_leaf_size)) + coord_offset;
        if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
            spdlog::warn("Voxel coord is out of range.");
            pair = {invalid_coord, i};
            return;
        }

        const std::uint64_t bits = (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
                                   (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
                                   (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
        pair = {bits, i};
    });

    std::sort(std::execution::par, coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<std::size_t> voxel_starts;
    voxel_starts.reserve(point_count + 1);
    voxel_starts.emplace_back(0);
    for (std::size_t i = 1; i < coord_pt.size(); ++i) {
        if (coord_pt[i].first != coord_pt[i - 1].first) {
            voxel_starts.emplace_back(i);
        }
    }
    voxel_starts.emplace_back(coord_pt.size());

    const std::size_t voxel_count = voxel_starts.empty() ? 0 : voxel_starts.size() - 1;
    std::vector<detail::VoxelAccumulator<Descriptor>> accumulators(voxel_count);

    std::for_each(
        std::execution::par,
        voxel_starts.begin(),
        voxel_starts.begin() + static_cast<std::ptrdiff_t>(voxel_count),
        [&](const std::size_t& start_idx) {
            const std::size_t voxel_idx = static_cast<std::size_t>(&start_idx - voxel_starts.data());
            const std::size_t end_idx = voxel_starts[voxel_idx + 1];

            if (coord_pt[start_idx].first == invalid_coord) {
                accumulators[voxel_idx].reset();
                return;
            }

            auto& accumulator = accumulators[voxel_idx];
            accumulator.reset();
            for (std::size_t i = start_idx; i < end_idx; ++i) {
                if (coord_pt[i].first == invalid_coord) {
                    continue;
                }
                accumulator.accumulate(*points, coord_pt[i].second);
            }
        });

    std::vector<detail::VoxelAccumulator<Descriptor>> valid_accums;
    valid_accums.reserve(accumulators.size());
    for (const auto& accum : accumulators) {
        if (accum.count > 0) {
            valid_accums.push_back(accum);
        }
    }

    auto downsampled = std::make_shared<Cloud>();
    downsampled->resize(valid_accums.size());
    for (std::size_t i = 0; i < valid_accums.size(); ++i) {
        detail::write_voxel_average(*downsampled, i, valid_accums[i]);
    }

    return downsampled;
}

/**
 * @brief 通用去畸变（点云）
 */
inline PointCloudType::Ptr DeskewPointCloud(const PointCloudType::ConstPtr& cloud, const PoseAtTimeFn& pose_query, const Eigen::Isometry3d& ref_pose, const Eigen::Isometry3d& T_i_l)
{
    if (!cloud) {
        spdlog::warn("DeskewPointCloud received null cloud");
        return PointCloudType::Ptr(new PointCloudType);
    }

    PointCloudType::Ptr deskewed_cloud = cloud->clone();
    if (cloud->empty()) {
        return deskewed_cloud;
    }

    std::vector<std::size_t> indices(cloud->size());
    std::iota(indices.begin(), indices.end(), 0);
    const Eigen::Isometry3f T_ref = (ref_pose * T_i_l).cast<float>();

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](std::size_t idx) {
        const double point_time = cloud->timestamp(idx);
        const auto pose_opt = pose_query(point_time);
        if (!pose_opt) {
            spdlog::error("Pose query failed for timestamp {:.6f}", point_time);
            return;
        }
        const Eigen::Isometry3f T_point = (pose_opt.value() * T_i_l).cast<float>();
        Eigen::Vector3f p = cloud->position(idx);
        p = T_ref.inverse() * T_point * p;
        deskewed_cloud->position(idx) = p;
    });

    return deskewed_cloud;
}

}  // namespace ms_slam::slam_core
