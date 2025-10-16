#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace ms_slam::slam_core
{

// ---------------------------------------------------------------------------
// Field tags and descriptors
// ---------------------------------------------------------------------------

struct PositionTag {};
struct IntensityTag {};
struct TimestampTag {};
struct RGBTag {};
struct NormalTag {};
struct CurvatureTag {};

template<typename ScalarT, typename TagT, std::size_t DimensionsV>
struct FieldDescriptor {
    using scalar_type = ScalarT;
    using tag = TagT;
    static constexpr std::size_t dimensions = DimensionsV;
};

template<typename ScalarT, typename... FieldDescriptors>
struct PointDescriptor {
    using scalar_type = ScalarT;
    using field_tuple = std::tuple<FieldDescriptors...>;
    static constexpr std::size_t field_count = sizeof...(FieldDescriptors);
};

template<typename Descriptor>
inline constexpr std::size_t field_count_v = Descriptor::field_count;

template<typename Tag, typename Descriptor>
struct has_field : std::false_type {};

template<typename Tag, typename ScalarT, typename... Fields>
struct has_field<Tag, PointDescriptor<ScalarT, Fields...>>
    : std::bool_constant<(std::is_same_v<Tag, typename Fields::tag> || ...)> {};

template<typename Tag, typename Descriptor>
inline constexpr bool has_field_v = has_field<Tag, Descriptor>::value;

namespace detail {

template<typename T>
struct dependent_false : std::false_type {};

inline constexpr std::size_t kInvalidIndex = static_cast<std::size_t>(-1);

template<typename Tag, typename Tuple>
struct field_index_impl;

template<typename Tag, typename... Fields>
struct field_index_impl<Tag, std::tuple<Fields...>> {
    static constexpr std::size_t value = []() consteval {
        constexpr std::array<bool, sizeof...(Fields)> matches{std::is_same_v<Tag, typename Fields::tag>...};
        for (std::size_t i = 0; i < matches.size(); ++i) {
            if (matches[i]) {
                return i;
            }
        }
        return kInvalidIndex;
    }();
};

}  // namespace detail

inline constexpr std::size_t kInvalidFieldIndex = detail::kInvalidIndex;

template<typename Tag, typename Descriptor>
inline constexpr std::size_t field_index_v = detail::field_index_impl<Tag, typename Descriptor::field_tuple>::value;

template<typename Tag, typename Descriptor>
using field_descriptor_t = std::tuple_element_t<field_index_v<Tag, Descriptor>, typename Descriptor::field_tuple>;

// ---------------------------------------------------------------------------
// Descriptor aliases
// ---------------------------------------------------------------------------

using PointXYZDescriptor = PointDescriptor<float, FieldDescriptor<float, PositionTag, 3>>;
using PointXYZIDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<float, IntensityTag, 1>>;
using PointXYZRGBDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<std::uint8_t, RGBTag, 3>>;
using PointXYZITDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<float, IntensityTag, 1>,
    FieldDescriptor<std::uint64_t, TimestampTag, 1>>;
using PointXYZIRGBDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<float, IntensityTag, 1>,
    FieldDescriptor<std::uint8_t, RGBTag, 3>>;
using PointXYZRGBTDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<std::uint8_t, RGBTag, 3>,
    FieldDescriptor<std::uint64_t, TimestampTag, 1>>;
using PointXYZINormalDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<float, IntensityTag, 1>,
    FieldDescriptor<float, NormalTag, 3>>;
using PointXYZINormalCurvatureDescriptor = PointDescriptor<float,
    FieldDescriptor<float, PositionTag, 3>,
    FieldDescriptor<float, IntensityTag, 1>,
    FieldDescriptor<float, NormalTag, 3>,
    FieldDescriptor<float, CurvatureTag, 1>>;

// ---------------------------------------------------------------------------
// Concrete point types (AoS convenience)
// ---------------------------------------------------------------------------

struct PointXYZ
{
    using Descriptor = PointXYZDescriptor;

    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    PointXYZ() = default;
    constexpr PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return std::array<float, 3>{x, y, z};
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZ");
        }
    }
};

struct PointXYZI : PointXYZ
{
    using Descriptor = PointXYZIDescriptor;

    float intensity{0.0f};

    PointXYZI() = default;
    constexpr PointXYZI(float x_, float y_, float z_, float intensity_)
        : PointXYZ(x_, y_, z_), intensity(intensity_) {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, IntensityTag>) {
            return intensity;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZI");
        }
    }
};

struct PointXYZRGB : PointXYZ
{
    using Descriptor = PointXYZRGBDescriptor;

    std::array<std::uint8_t, 3> rgb{0, 0, 0};

    PointXYZRGB() = default;
    constexpr PointXYZRGB(float x_, float y_, float z_, std::uint8_t r, std::uint8_t g, std::uint8_t b)
        : PointXYZ(x_, y_, z_), rgb{r, g, b} {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, RGBTag>) {
            return rgb;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZRGB");
        }
    }
};

struct PointXYZIT : PointXYZ
{
    using Descriptor = PointXYZITDescriptor;

    float intensity{0.0f};
    std::uint64_t timestamp{0ULL};

    PointXYZIT() = default;
    constexpr PointXYZIT(float x_, float y_, float z_, float intensity_, std::uint64_t timestamp_)
        : PointXYZ(x_, y_, z_), intensity(intensity_), timestamp(timestamp_) {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, IntensityTag>) {
            return intensity;
        } else if constexpr (std::is_same_v<Tag, TimestampTag>) {
            return timestamp;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZIT");
        }
    }
};

struct PointXYZIRGB : PointXYZ
{
    using Descriptor = PointXYZIRGBDescriptor;

    float intensity{0.0f};
    std::array<std::uint8_t, 3> rgb{0, 0, 0};

    PointXYZIRGB() = default;
    constexpr PointXYZIRGB(float x_, float y_, float z_, float intensity_, std::uint8_t r, std::uint8_t g, std::uint8_t b)
        : PointXYZ(x_, y_, z_), intensity(intensity_), rgb{r, g, b} {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, IntensityTag>) {
            return intensity;
        } else if constexpr (std::is_same_v<Tag, RGBTag>) {
            return rgb;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZIRGB");
        }
    }
};

struct PointXYZRGBT : PointXYZ
{
    using Descriptor = PointXYZRGBTDescriptor;

    std::array<std::uint8_t, 3> rgb{0, 0, 0};
    std::uint64_t timestamp{0ULL};

    PointXYZRGBT() = default;
    constexpr PointXYZRGBT(float x_, float y_, float z_, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint64_t timestamp_)
        : PointXYZ(x_, y_, z_), rgb{r, g, b}, timestamp(timestamp_) {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, RGBTag>) {
            return rgb;
        } else if constexpr (std::is_same_v<Tag, TimestampTag>) {
            return timestamp;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZRGBT");
        }
    }
};

struct PointXYZINormal : PointXYZ
{
    using Descriptor = PointXYZINormalDescriptor;

    float intensity{0.0f};
    std::array<float, 3> normal{0.0f, 0.0f, 1.0f};

    PointXYZINormal() = default;
    constexpr PointXYZINormal(float x_, float y_, float z_, float intensity_, float nx, float ny, float nz)
        : PointXYZ(x_, y_, z_), intensity(intensity_), normal{nx, ny, nz} {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, PositionTag>) {
            return PointXYZ::template get<Tag>();
        } else if constexpr (std::is_same_v<Tag, IntensityTag>) {
            return intensity;
        } else if constexpr (std::is_same_v<Tag, NormalTag>) {
            return normal;
        } else {
            static_assert(detail::dependent_false<Tag>::value, "Unsupported tag for PointXYZINormal");
        }
    }
};

struct PointXYZINormalCurvature : PointXYZINormal
{
    using Descriptor = PointXYZINormalCurvatureDescriptor;

    float curvature{0.0f};

    PointXYZINormalCurvature() = default;
    constexpr PointXYZINormalCurvature(float x_, float y_, float z_, float intensity_, float nx, float ny, float nz, float curvature_)
        : PointXYZINormal(x_, y_, z_, intensity_, nx, ny, nz), curvature(curvature_) {}

    template<typename Tag>
    auto get() const
    {
        if constexpr (std::is_same_v<Tag, CurvatureTag>) {
            return curvature;
        } else {
            return PointXYZINormal::template get<Tag>();
        }
    }
};

}  // namespace ms_slam::slam_core
