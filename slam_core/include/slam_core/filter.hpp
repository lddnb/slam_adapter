/**
 * @file filter.hpp
 * @brief 点云降采样工具
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "slam_core/point_cloud.hpp"

namespace ms_slam::slam_core
{

inline Eigen::Array3i fast_floor(const Eigen::Array3f& pt)
{
    const Eigen::Array3i ncoord = pt.cast<int>();
    return ncoord - (pt < ncoord.cast<float>()).cast<int>();
}

template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp_impl(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp)
{
    const std::ptrdiff_t n = std::distance(first, last);
    if (n < 1024) {
        std::sort(first, last, comp);
        return;
    }

    const auto median3 = [&](const auto& a, const auto& b, const auto& c, const Compare& cmp) {
        return cmp(a, b) ? (cmp(b, c) ? b : (cmp(a, c) ? c : a)) : (cmp(a, c) ? a : (cmp(b, c) ? c : b));
    };

    const int offset = n / 8;
    const auto m1 = median3(*first, *(first + offset), *(first + offset * 2), comp);
    const auto m2 = median3(*(first + offset * 3), *(first + offset * 4), *(first + offset * 5), comp);
    const auto m3 = median3(*(first + offset * 6), *(first + offset * 7), *(last - 1), comp);

    auto pivot = median3(m1, m2, m3, comp);
    auto middle1 = std::partition(first, last, [&](const auto& val) { return comp(val, pivot); });
    auto middle2 = std::partition(middle1, last, [&](const auto& val) { return !comp(pivot, val); });

#pragma omp task
    quick_sort_omp_impl(first, middle1, comp);

#pragma omp task
    quick_sort_omp_impl(middle2, last, comp);
}

template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads)
{
#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp single nowait
        {
            quick_sort_omp_impl(first, last, comp);
        }
    }
#else
    std::sort(first, last, comp);
#endif
}

namespace detail
{

template <typename Descriptor>
struct VoxelAccumulator {
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

    template <typename Cloud>
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

template <typename Descriptor>
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

template <typename Descriptor>
typename PointCloud<Descriptor>::Ptr
voxelgrid_sampling_omp(const typename PointCloud<Descriptor>::ConstPtr& points, double leaf_size, int num_threads = 4)
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

#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(point_count); ++i) {
        const auto pos = points->position(static_cast<std::size_t>(i));
        const Eigen::Array3f pt_f = pos.array();
        const Eigen::Array3i coord = fast_floor(pt_f * static_cast<float>(inv_leaf_size)) + coord_offset;
        if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
            std::cerr << "warning: voxel coord is out of range!!" << std::endl;
            coord_pt[static_cast<std::size_t>(i)] = {invalid_coord, static_cast<std::size_t>(i)};
            continue;
        }

        const std::uint64_t bits = (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |
                                   (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |
                                   (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
        coord_pt[static_cast<std::size_t>(i)] = {bits, static_cast<std::size_t>(i)};
    }

    quick_sort_omp(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }, num_threads);

    auto downsampled = std::make_shared<Cloud>();
    downsampled->resize(point_count);

    const int block_size = 1024;
    std::atomic_uint64_t num_points = 0;

#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
    for (std::int64_t block_begin = 0; block_begin < static_cast<std::int64_t>(point_count); block_begin += block_size) {
        const std::size_t block_end = std::min<std::size_t>(point_count, static_cast<std::size_t>(block_begin + block_size));

        std::size_t first = static_cast<std::size_t>(block_begin);
        while (first < block_end && coord_pt[first].first == invalid_coord) {
            ++first;
        }

        if (first >= block_end) {
            continue;
        }

        std::vector<detail::VoxelAccumulator<Descriptor>> sub_accums;
        sub_accums.reserve(block_size);

        detail::VoxelAccumulator<Descriptor> accum;
        accum.reset();
        accum.accumulate(*points, coord_pt[first].second);
        std::uint64_t current_key = coord_pt[first].first;

        for (std::size_t i = first + 1; i < block_end; ++i) {
            const auto key = coord_pt[i].first;
            if (key == invalid_coord) {
                continue;
            }

            if (key != current_key) {
                sub_accums.push_back(accum);
                accum.reset();
                accum.accumulate(*points, coord_pt[i].second);
                current_key = key;
            } else {
                accum.accumulate(*points, coord_pt[i].second);
            }
        }

        sub_accums.push_back(accum);

        const std::size_t point_index_begin = num_points.fetch_add(sub_accums.size());
        for (std::size_t i = 0; i < sub_accums.size(); ++i) {
            detail::write_voxel_average(*downsampled, point_index_begin + i, sub_accums[i]);
        }
    }

    downsampled->resize(num_points.load());
    return downsampled;
}

template <typename Descriptor>
typename PointCloud<Descriptor>::Ptr voxelgrid_sampling_pstl(const typename PointCloud<Descriptor>::ConstPtr& points, double leaf_size)
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
        const Eigen::Array3i coord = fast_floor(pt_f * static_cast<float>(inv_leaf_size)) + coord_offset;
        if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
            std::cerr << "warning: voxel coord is out of range!!" << std::endl;
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

    const std::size_t voxel_count = voxel_starts.size() > 0 ? voxel_starts.size() - 1 : 0;
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

}  // namespace ms_slam::slam_core
