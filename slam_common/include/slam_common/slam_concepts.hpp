#pragma once

#include <concepts>
#include <ranges>
#include <cstdint>

namespace ms_slam::slam_common
{

// C++20 Concepts for Point Types
template <typename T>
concept BasicPoint = requires(T point) {
    { point.x } -> std::convertible_to<float>;
    { point.y } -> std::convertible_to<float>;
    { point.z } -> std::convertible_to<float>;
};

template <typename T>
concept HasTimestamp = BasicPoint<T> && requires(T point) {
    { point.timestamp } -> std::convertible_to<uint64_t>;
};

template <typename T>
concept HasIntensity = BasicPoint<T> && requires(T point) {
    { point.intensity } -> std::convertible_to<float>;
};

template <typename T>
concept HasColor = BasicPoint<T> && requires(T point) {
    { point.R } -> std::convertible_to<uint8_t>;
    { point.G } -> std::convertible_to<uint8_t>;
    { point.B } -> std::convertible_to<uint8_t>;
};

template <typename T>
concept FullFeaturedPoint = HasTimestamp<T> && HasIntensity<T> && HasColor<T>;

// Concept for PointCloud types
template <typename T>
concept IsPointCloud = requires(T cloud) {
    typename T::value_type;
    { cloud.points } -> std::ranges::range;
    { cloud.timestamp } -> std::convertible_to<uint64_t>;
    { cloud.seq } -> std::convertible_to<uint32_t>;
    { cloud.num_points } -> std::convertible_to<uint32_t>;
} && BasicPoint<typename T::value_type>;

template <typename T>
concept Serializable = requires { T::IOX2_TYPE_NAME; };

template <typename T>
concept Message = Serializable<T> && requires(T msg) {
    { msg.timestamp } -> std::convertible_to<uint64_t>;
    { msg.seq } -> std::convertible_to<uint32_t>;
};

// Consteval helper functions for compile-time calculations
template <typename T>
consteval bool supports_timestamp()
{
    return HasTimestamp<T>;
}

template <typename T>
consteval bool supports_intensity()
{
    return HasIntensity<T>;
}

template <typename T>
consteval bool supports_color()
{
    return HasColor<T>;
}

// Compile-time point type information
template <typename T>
constexpr size_t point_field_count()
{
    size_t count = 3;  // x, y, z
    if constexpr (HasTimestamp<T>) count++;
    if constexpr (HasIntensity<T>) count++;
    if constexpr (HasColor<T>) count += 3;  // R, G, B
    return count;
}

// Constinit for compile-time constants
template <typename T>
constinit const bool has_timestamp_v = HasTimestamp<T>;

template <typename T>
constinit const bool has_intensity_v = HasIntensity<T>;

template <typename T>
constinit const bool has_color_v = HasColor<T>;

// Optimized point field setters using fold expressions
template <BasicPoint PointType>
constexpr void set_basic_fields(PointType& point, float x, float y, float z)
{
    point.x = x;
    point.y = y;
    point.z = z;
}

template <HasTimestamp PointType>
constexpr void set_timestamp(PointType& point, uint64_t timestamp)
{
    point.timestamp = timestamp;
}

template <HasIntensity PointType>
constexpr void set_intensity(PointType& point, float intensity)
{
    point.intensity = intensity;
}

template <HasColor PointType>
constexpr void set_color(PointType& point, uint8_t r, uint8_t g, uint8_t b)
{
    point.R = r;
    point.G = g;
    point.B = b;
}

// Advanced C++20 template features with fold expressions
template <BasicPoint... PointTypes>
constexpr auto get_max_field_count()
{
    return std::max({point_field_count<PointTypes>()...});
}

// Variadic template for multiple point operations using fold expressions
template <BasicPoint... PointTypes>
constexpr bool all_have_timestamp()
{
    return (HasTimestamp<PointTypes> && ...);
}

template <BasicPoint... PointTypes>
constexpr bool any_have_color()
{
    return (HasColor<PointTypes> || ...);
}

// Template parameter pack for batch point processing
template <typename... Points>
    requires(BasicPoint<Points> && ...)
constexpr auto process_multiple_points(Points&&... points)
{
    return std::make_tuple(std::forward<Points>(points)...);
}

// C++20 ranges-based point cloud utilities
template <IsPointCloud CloudType>
auto get_valid_points(const CloudType& cloud)
{
    return cloud.points | std::views::filter([](const auto& point) { return point.x != 0.0f || point.y != 0.0f || point.z != 0.0f; });
}

template <IsPointCloud CloudType>
auto get_points_by_distance(const CloudType& cloud, float max_distance)
{
    return cloud.points | std::views::filter([max_distance](const auto& point) {
               float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
               return distance <= max_distance;
           });
}

}  // namespace ms_slam::slam_common