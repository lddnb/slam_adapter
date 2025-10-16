#pragma once

#include "point_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ms_slam::slam_core
{

namespace detail
{

template<typename T>
struct aligned_vector {
    using type = std::conditional_t<std::is_floating_point_v<T>,
        std::vector<T, Eigen::aligned_allocator<T>>,
        std::vector<T>>;
};

template<typename T>
using aligned_vector_t = typename aligned_vector<T>::type;

template<typename Descriptor, std::size_t... Indices>
constexpr auto make_storage_tuple(std::index_sequence<Indices...>)
{
    using field_tuple = typename Descriptor::field_tuple;
    return std::tuple<aligned_vector_t<typename std::tuple_element_t<Indices, field_tuple>::scalar_type>...>{};
}

}  // namespace detail

template<typename Descriptor>
class PointCloud
{
public:
    using descriptor_type = Descriptor;
    using scalar_type = typename descriptor_type::scalar_type;

    static_assert(has_field_v<PositionTag, descriptor_type>,
        "PointCloud requires a PositionTag field");

    static constexpr std::size_t field_count = field_count_v<descriptor_type>;

    using PositionDescriptor = field_descriptor_t<PositionTag, descriptor_type>;
    static constexpr std::size_t position_dimensions = PositionDescriptor::dimensions;
    static_assert(position_dimensions == 3, "Current implementation assumes 3D positions");

    using PositionScalar = typename PositionDescriptor::scalar_type;
    static_assert(std::is_same_v<PositionScalar, scalar_type>,
        "Position scalar must match descriptor scalar type");

    using PositionMatrixMap = Eigen::Map<Eigen::Matrix<scalar_type, 3, Eigen::Dynamic, Eigen::ColMajor>>;
    using ConstPositionMatrixMap = Eigen::Map<const Eigen::Matrix<scalar_type, 3, Eigen::Dynamic, Eigen::ColMajor>>;

    PointCloud() = default;

    explicit PointCloud(std::size_t initial_size)
    {
        resize(initial_size);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    void reserve(std::size_t new_capacity)
    {
        if (new_capacity <= capacity_) {
            return;
        }
        reserve_fields(new_capacity);
        capacity_ = new_capacity;
    }

    void resize(std::size_t new_size)
    {
        ensure_capacity(new_size);
        resize_fields(new_size);
        size_ = new_size;
    }

    void clear() noexcept
    {
        resize_fields(0);
        size_ = 0;
    }

    void release() noexcept
    {
        size_ = 0;
        capacity_ = 0;
        release_fields();
    }

    void push_back(scalar_type x, scalar_type y, scalar_type z)
    {
        ensure_capacity(size_ + 1);
        auto& pos = storage_by_tag<PositionTag>();
        pos.resize((size_ + 1) * position_dimensions);
        pos[size_ * position_dimensions + 0] = x;
        pos[size_ * position_dimensions + 1] = y;
        pos[size_ * position_dimensions + 2] = z;
        expand_optional_fields(size_ + 1);
        ++size_;
    }

    template<typename Derived>
    void push_back(const Eigen::MatrixBase<Derived>& position)
    {
        static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1,
            "push_back expects a 3x1 vector");
        Eigen::Matrix<scalar_type, 3, 1> casted = position.template cast<scalar_type>();
        push_back(casted.x(), casted.y(), casted.z());
    }

    template<typename PointT>
    void push_back(const PointT& point)
    {
        ensure_capacity(size_ + 1);
        auto& pos = storage_by_tag<PositionTag>();
        const std::size_t index = size_;
        pos.resize((index + 1) * position_dimensions);
        expand_optional_fields(index + 1);

        assign_position(point, index);
        copy_optional_fields(point, index);

        size_ = index + 1;
    }

    template<std::ranges::input_range Range>
    void append(const Range& points)
    {
        if constexpr (std::ranges::sized_range<Range>) {
            ensure_capacity(size_ + static_cast<std::size_t>(std::ranges::size(points)));
        }
        for (const auto& point : points) {
            push_back(point);
        }
    }

    template<typename PointT>
    void append(std::initializer_list<PointT> points)
    {
        append(std::span(points.begin(), points.end()));
    }

    [[nodiscard]] std::span<scalar_type> positions() noexcept
    {
        auto& pos = storage_by_tag<PositionTag>();
        return std::span<scalar_type>(pos.data(), size_ * position_dimensions);
    }

    [[nodiscard]] std::span<const scalar_type> positions() const noexcept
    {
        const auto& pos = storage_by_tag<PositionTag>();
        return std::span<const scalar_type>(pos.data(), size_ * position_dimensions);
    }

    [[nodiscard]] PositionMatrixMap positions_matrix() noexcept
    {
        auto& pos = storage_by_tag<PositionTag>();
        return PositionMatrixMap(pos.data(), 3, static_cast<Eigen::Index>(size_));
    }

    [[nodiscard]] ConstPositionMatrixMap positions_matrix() const noexcept
    {
        const auto& pos = storage_by_tag<PositionTag>();
        return ConstPositionMatrixMap(pos.data(), 3, static_cast<Eigen::Index>(size_));
    }

    [[nodiscard]] Eigen::Map<Eigen::Matrix<scalar_type, 3, 1>> position(std::size_t index) noexcept
    {
        assert(index < size_);
        auto& pos = storage_by_tag<PositionTag>();
        return Eigen::Map<Eigen::Matrix<scalar_type, 3, 1>>(pos.data() + index * position_dimensions);
    }

    [[nodiscard]] Eigen::Map<const Eigen::Matrix<scalar_type, 3, 1>> position(std::size_t index) const noexcept
    {
        assert(index < size_);
        const auto& pos = storage_by_tag<PositionTag>();
        return Eigen::Map<const Eigen::Matrix<scalar_type, 3, 1>>(pos.data() + index * position_dimensions);
    }

    template<typename Tag>
        requires has_field_v<Tag, descriptor_type> && (!std::is_same_v<Tag, PositionTag>)
    [[nodiscard]] auto field_block(std::size_t index) noexcept
    {
        assert(index < size_);
        using FieldDesc = field_descriptor_t<Tag, descriptor_type>;
        auto& vec = storage_by_tag<Tag>();
        return std::span<typename FieldDesc::scalar_type>(vec.data() + index * FieldDesc::dimensions, FieldDesc::dimensions);
    }

    template<typename Tag>
        requires has_field_v<Tag, descriptor_type> && (!std::is_same_v<Tag, PositionTag>)
    [[nodiscard]] auto field_block(std::size_t index) const noexcept
    {
        assert(index < size_);
        using FieldDesc = field_descriptor_t<Tag, descriptor_type>;
        const auto& vec = storage_by_tag<Tag>();
        return std::span<const typename FieldDesc::scalar_type>(vec.data() + index * FieldDesc::dimensions, FieldDesc::dimensions);
    }

    template<typename Tag>
    [[nodiscard]] auto field_span() noexcept
    {
        static_assert(has_field_v<Tag, descriptor_type>, "Requested field is not part of the descriptor");
        using FieldDesc = field_descriptor_t<Tag, descriptor_type>;
        auto& vec = storage_by_tag<Tag>();
        return std::span<typename FieldDesc::scalar_type>(vec.data(), size_ * FieldDesc::dimensions);
    }

    template<typename Tag>
    [[nodiscard]] auto field_span() const noexcept
    {
        static_assert(has_field_v<Tag, descriptor_type>, "Requested field is not part of the descriptor");
        using FieldDesc = field_descriptor_t<Tag, descriptor_type>;
        const auto& vec = storage_by_tag<Tag>();
        return std::span<const typename FieldDesc::scalar_type>(vec.data(), size_ * FieldDesc::dimensions);
    }

    template<typename TransformType>
        requires requires(const TransformType& t) { t.linear(); t.translation(); }
    PointCloud& transform(const TransformType& transform) noexcept
    {
        apply_transform(transform.linear(), transform.translation());
        return *this;
    }

    template<typename TransformType>
        requires requires(const TransformType& t) { t.linear(); t.translation(); }
    PointCloud& transform_parallel(const TransformType& transform) noexcept
    {
        apply_transform_parallel(transform.linear(), transform.translation());
        return *this;
    }

    template<typename TransformType>
        requires requires(const TransformType& t) { t.linear(); t.translation(); }
    [[nodiscard]] PointCloud transformed(const TransformType& transform) const
    {
        PointCloud copy(*this);
        copy.transform(transform);
        return copy;
    }

    template<typename TransformType>
        requires requires(const TransformType& t) { t.linear(); t.translation(); }
    [[nodiscard]] PointCloud transformed_parallel(const TransformType& transform) const
    {
        PointCloud copy(*this);
        copy.apply_transform_parallel(transform);
        return copy;
    }

    template<typename Matrix4>
    PointCloud& transform_matrix(const Matrix4& homogeneous) noexcept
    {
        assert(homogeneous.rows() == 4 && homogeneous.cols() == 4);
        Eigen::Matrix<scalar_type, 3, 3> R = homogeneous.template block<3, 3>(0, 0).template cast<scalar_type>();
        Eigen::Matrix<scalar_type, 3, 1> t = homogeneous.template block<3, 1>(0, 3).template cast<scalar_type>();
        apply_transform(R, t);
        return *this;
    }

    template<typename Matrix3, typename Vector3>
    PointCloud& transform_linear(const Matrix3& R, const Vector3& t) noexcept
    {
        apply_transform(R, t);
        return *this;
    }

    template<typename UnaryFn>
    void for_each_position(UnaryFn&& fn)
    {
        auto data = storage_by_tag<PositionTag>().data();
        for (std::size_t i = 0; i < size_; ++i) {
            Eigen::Map<Eigen::Matrix<scalar_type, 3, 1>> point(data + i * position_dimensions);
            fn(point);
        }
    }

private:
    using storage_tuple_t = decltype(detail::make_storage_tuple<descriptor_type>(std::make_index_sequence<field_count>{}));

    storage_tuple_t storage_{};
    std::size_t size_{0};
    std::size_t capacity_{0};

    template<typename Tag>
    auto& storage_by_tag() noexcept
    {
        constexpr std::size_t idx = field_index_v<Tag, descriptor_type>;
        static_assert(idx != kInvalidFieldIndex, "Requested field not available");
        return std::get<idx>(storage_);
    }

    template<typename Tag>
    const auto& storage_by_tag() const noexcept
    {
        constexpr std::size_t idx = field_index_v<Tag, descriptor_type>;
        static_assert(idx != kInvalidFieldIndex, "Requested field not available");
        return std::get<idx>(storage_);
    }

    template<typename Matrix3, typename Vector3>
    void apply_transform(const Matrix3& rotation, const Vector3& translation)
    {
        if (size_ == 0) {
            return;
        }

        const Eigen::Matrix<scalar_type, 3, 3> R = rotation.template cast<scalar_type>();
        const Eigen::Matrix<scalar_type, 3, 1> t = translation.template cast<scalar_type>();

        const scalar_type r00 = R(0, 0);
        const scalar_type r01 = R(0, 1);
        const scalar_type r02 = R(0, 2);
        const scalar_type r10 = R(1, 0);
        const scalar_type r11 = R(1, 1);
        const scalar_type r12 = R(1, 2);
        const scalar_type r20 = R(2, 0);
        const scalar_type r21 = R(2, 1);
        const scalar_type r22 = R(2, 2);

        const scalar_type tx = t.x();
        const scalar_type ty = t.y();
        const scalar_type tz = t.z();

        scalar_type* pos_ptr = storage_by_tag<PositionTag>().data();
        for (std::size_t i = 0; i < size_; ++i, pos_ptr += position_dimensions) {
            const scalar_type x = pos_ptr[0];
            const scalar_type y = pos_ptr[1];
            const scalar_type z = pos_ptr[2];

            pos_ptr[0] = r00 * x + r01 * y + r02 * z + tx;
            pos_ptr[1] = r10 * x + r11 * y + r12 * z + ty;
            pos_ptr[2] = r20 * x + r21 * y + r22 * z + tz;
        }

        if constexpr (has_field_v<NormalTag, descriptor_type>) {
            scalar_type* normal_ptr = storage_by_tag<NormalTag>().data();
            for (std::size_t i = 0; i < size_; ++i, normal_ptr += 3) {
                const scalar_type nx = normal_ptr[0];
                const scalar_type ny = normal_ptr[1];
                const scalar_type nz = normal_ptr[2];

                normal_ptr[0] = r00 * nx + r01 * ny + r02 * nz;
                normal_ptr[1] = r10 * nx + r11 * ny + r12 * nz;
                normal_ptr[2] = r20 * nx + r21 * ny + r22 * nz;
            }
        }
    }
    
    template<typename Matrix3, typename Vector3>
    void apply_transform_parallel(const Matrix3& rotation, const Vector3& translation)
    {
        const Eigen::Matrix<scalar_type, 3, 3> R = rotation.template cast<scalar_type>();
        const Eigen::Matrix<scalar_type, 3, 1> t = translation.template cast<scalar_type>();

        if (size_ == 0) {
            return;
        }

        const scalar_type r00 = R(0, 0);
        const scalar_type r01 = R(0, 1);
        const scalar_type r02 = R(0, 2);
        const scalar_type r10 = R(1, 0);
        const scalar_type r11 = R(1, 1);
        const scalar_type r12 = R(1, 2);
        const scalar_type r20 = R(2, 0);
        const scalar_type r21 = R(2, 1);
        const scalar_type r22 = R(2, 2);

        const scalar_type tx = t.x();
        const scalar_type ty = t.y();
        const scalar_type tz = t.z();

        scalar_type* pos_base = storage_by_tag<PositionTag>().data();

        if constexpr (has_field_v<NormalTag, descriptor_type>) {
            scalar_type* normal_base = storage_by_tag<NormalTag>().data();
            const auto indices = std::views::iota(std::size_t{0}, size_);
            std::for_each(std::execution::par, indices.begin(), indices.end(), [=](std::size_t index) {
                scalar_type* pos = pos_base + index * position_dimensions;
                const scalar_type x = pos[0];
                const scalar_type y = pos[1];
                const scalar_type z = pos[2];

                pos[0] = r00 * x + r01 * y + r02 * z + tx;
                pos[1] = r10 * x + r11 * y + r12 * z + ty;
                pos[2] = r20 * x + r21 * y + r22 * z + tz;

                scalar_type* normal = normal_base + index * 3;
                const scalar_type nx = normal[0];
                const scalar_type ny = normal[1];
                const scalar_type nz = normal[2];

                normal[0] = r00 * nx + r01 * ny + r02 * nz;
                normal[1] = r10 * nx + r11 * ny + r12 * nz;
                normal[2] = r20 * nx + r21 * ny + r22 * nz;
            });
        } else {
            const auto indices = std::views::iota(std::size_t{0}, size_);
            std::for_each(std::execution::par, indices.begin(), indices.end(), [=](std::size_t index) {
                scalar_type* pos = pos_base + index * position_dimensions;
                const scalar_type x = pos[0];
                const scalar_type y = pos[1];
                const scalar_type z = pos[2];

                pos[0] = r00 * x + r01 * y + r02 * z + tx;
                pos[1] = r10 * x + r11 * y + r12 * z + ty;
                pos[2] = r20 * x + r21 * y + r22 * z + tz;
            });
        }
    }

    void ensure_capacity(std::size_t desired)
    {
        if (desired <= capacity_) {
            return;
        }
        std::size_t new_capacity = std::max<std::size_t>(desired, capacity_ == 0 ? 8 : capacity_ * 2);
        reserve(new_capacity);
    }

    template<std::size_t... Indices>
    void reserve_fields_impl(std::size_t new_capacity, std::index_sequence<Indices...>)
    {
        (std::get<Indices>(storage_).reserve(new_capacity * field_dimensions_by_index<Indices>()), ...);
    }

    void reserve_fields(std::size_t new_capacity)
    {
        reserve_fields_impl(new_capacity, std::make_index_sequence<field_count>{});
    }

    template<std::size_t... Indices>
    void resize_fields_impl(std::size_t new_size, std::index_sequence<Indices...>)
    {
        (std::get<Indices>(storage_).resize(new_size * field_dimensions_by_index<Indices>()), ...);
    }

    void resize_fields(std::size_t new_size)
    {
        resize_fields_impl(new_size, std::make_index_sequence<field_count>{});
    }

    void expand_optional_fields(std::size_t new_size)
    {
        expand_optional_fields_impl(new_size, std::make_index_sequence<field_count>{});
    }

    template<std::size_t... Indices>
    void expand_optional_fields_impl(std::size_t new_size, std::index_sequence<Indices...>)
    {
        (expand_single_field<Indices>(new_size), ...);
    }

    template<std::size_t Index>
    void expand_single_field(std::size_t new_size)
    {
        using Field = std::tuple_element_t<Index, typename descriptor_type::field_tuple>;
        constexpr bool is_position = std::is_same_v<typename Field::tag, PositionTag>;
        if constexpr (!is_position) {
            auto& vec = std::get<Index>(storage_);
            vec.resize(new_size * Field::dimensions);
        }
    }

    template<std::size_t Index>
    static constexpr std::size_t field_dimensions_by_index()
    {
        using Field = std::tuple_element_t<Index, typename descriptor_type::field_tuple>;
        return Field::dimensions;
    }

    template<typename PointT>
    void assign_position(const PointT& point, std::size_t index)
    {
        auto& pos = storage_by_tag<PositionTag>();
        auto set_xyz = [&](scalar_type x, scalar_type y, scalar_type z) {
            pos[index * position_dimensions + 0] = x;
            pos[index * position_dimensions + 1] = y;
            pos[index * position_dimensions + 2] = z;
        };

        if constexpr (requires { point.template get<PositionTag>(); }) {
            const auto value = point.template get<PositionTag>();
            set_xyz(static_cast<scalar_type>(value[0]),
                    static_cast<scalar_type>(value[1]),
                    static_cast<scalar_type>(value[2]));
        } else if constexpr (requires { point.position; }) {
            set_xyz(static_cast<scalar_type>(point.position[0]),
                    static_cast<scalar_type>(point.position[1]),
                    static_cast<scalar_type>(point.position[2]));
        } else if constexpr (requires { point.x; point.y; point.z; }) {
            set_xyz(static_cast<scalar_type>(point.x),
                    static_cast<scalar_type>(point.y),
                    static_cast<scalar_type>(point.z));
        } else if constexpr (requires { point[0]; point[1]; point[2]; }) {
            set_xyz(static_cast<scalar_type>(point[0]),
                    static_cast<scalar_type>(point[1]),
                    static_cast<scalar_type>(point[2]));
        } else if constexpr (requires { point(0); point(1); point(2); }) {
            set_xyz(static_cast<scalar_type>(point(0)),
                    static_cast<scalar_type>(point(1)),
                    static_cast<scalar_type>(point(2)));
        } else if constexpr (requires { point.x(); point.y(); point.z(); }) {
            set_xyz(static_cast<scalar_type>(point.x()),
                    static_cast<scalar_type>(point.y()),
                    static_cast<scalar_type>(point.z()));
        } else {
            static_assert(detail::dependent_false<PointT>::value, "Point type does not provide position data");
        }
    }

    template<typename PointT>
    void copy_optional_fields(const PointT& point, std::size_t index)
    {
        copy_optional_fields_impl(point, index, std::make_index_sequence<field_count>{});
    }

    template<typename PointT, std::size_t... Indices>
    void copy_optional_fields_impl(const PointT& point, std::size_t index, std::index_sequence<Indices...>)
    {
        (copy_single_field<Indices>(point, index), ...);
    }

    template<std::size_t Index, typename PointT>
    void copy_single_field(const PointT& point, std::size_t index)
    {
        using Field = std::tuple_element_t<Index, typename descriptor_type::field_tuple>;
        if constexpr (!std::is_same_v<typename Field::tag, PositionTag>) {
            auto& storage_vec = std::get<Index>(storage_);
            using FieldScalar = typename Field::scalar_type;
            constexpr std::size_t dims = Field::dimensions;

            if constexpr (dims == 1) {
                if constexpr (requires { point.template get<typename Field::tag>(); }) {
                    storage_vec[index] = static_cast<FieldScalar>(point.template get<typename Field::tag>());
                } else if constexpr (std::is_same_v<typename Field::tag, IntensityTag> && requires { point.intensity; }) {
                    storage_vec[index] = static_cast<FieldScalar>(point.intensity);
                } else if constexpr (std::is_same_v<typename Field::tag, TimestampTag> && requires { point.timestamp; }) {
                    storage_vec[index] = static_cast<FieldScalar>(point.timestamp);
                } else if constexpr (std::is_same_v<typename Field::tag, CurvatureTag> && requires { point.curvature; }) {
                    storage_vec[index] = static_cast<FieldScalar>(point.curvature);
                } else {
                    static_assert(detail::dependent_false<PointT>::value, "Unsupported scalar field for push_back");
                }
            } else {
                FieldScalar* dst = storage_vec.data() + index * dims;
                if constexpr (requires { point.template get<typename Field::tag>(); }) {
                    auto field_value = point.template get<typename Field::tag>();
                    for (std::size_t i = 0; i < dims; ++i) {
                        dst[i] = static_cast<FieldScalar>(field_value[i]);
                    }
                } else if constexpr (std::is_same_v<typename Field::tag, NormalTag> && requires { point.normal; }) {
                    for (std::size_t i = 0; i < dims; ++i) {
                        dst[i] = static_cast<FieldScalar>(point.normal[i]);
                    }
                } else if constexpr (std::is_same_v<typename Field::tag, RGBTag> && requires { point.rgb; }) {
                    for (std::size_t i = 0; i < dims; ++i) {
                        dst[i] = static_cast<FieldScalar>(point.rgb[i]);
                    }
                } else {
                    static_assert(detail::dependent_false<PointT>::value, "Unsupported vector field for push_back");
                }
            }
        }
    }

    void release_fields() noexcept
    {
        release_fields_impl(std::make_index_sequence<field_count>{});
    }

    template<std::size_t... Indices>
    void release_fields_impl(std::index_sequence<Indices...>) noexcept
    {
        (release_single_field<Indices>(), ...);
    }

    template<std::size_t Index>
    void release_single_field() noexcept
    {
        auto& vec = std::get<Index>(storage_);
        vec.clear();
        vec.shrink_to_fit();
    }
};

using PointCloudXYZ = PointCloud<PointXYZDescriptor>;
using PointCloudXYZI = PointCloud<PointXYZIDescriptor>;
using PointCloudXYZRGB = PointCloud<PointXYZRGBDescriptor>;
using PointCloudXYZIT = PointCloud<PointXYZITDescriptor>;
using PointCloudXYZIRGB = PointCloud<PointXYZIRGBDescriptor>;
using PointCloudXYZRGBT = PointCloud<PointXYZRGBTDescriptor>;
using PointCloudXYZINormal = PointCloud<PointXYZINormalDescriptor>;
using PointCloudXYZINormalCurvature = PointCloud<PointXYZINormalCurvatureDescriptor>;

}  // namespace ms_slam::slam_core
