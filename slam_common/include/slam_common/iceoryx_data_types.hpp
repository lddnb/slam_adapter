#pragma once

#include <cstdint>
#include <iostream>

#include <iox2/iceoryx2.hpp>

namespace ms_slam::slam_common
{
// 各种不同类型的点
struct PointI {
    static constexpr const char* IOX2_TYPE_NAME = "PointI";

    float x;          // X coordinate in meters
    float y;          // Y coordinate in meters
    float z;          // Z coordinate in meters
    float intensity;  // Reflectivity as percentage
};

struct PointIT {
    static constexpr const char* IOX2_TYPE_NAME = "PointIT";

    float x;             // X coordinate in meters
    float y;             // Y coordinate in meters
    float z;             // Z coordinate in meters
    float intensity;     // Reflectivity as percentage
    uint64_t timestamp;  // UTC timestamp
};

struct PointRGB {
    static constexpr const char* IOX2_TYPE_NAME = "PointRGB";

    float x;    // X coordinate in meters
    float y;    // Y coordinate in meters
    float z;    // Z coordinate in meters
    uint8_t R;  // Color Red
    uint8_t G;  // Color Green
    uint8_t B;  // Color blue
};

struct PointRGBT {
    static constexpr const char* IOX2_TYPE_NAME = "PointRGBT";

    float x;             // X coordinate in meters
    float y;             // Y coordinate in meters
    float z;             // Z coordinate in meters
    uint8_t R;           // Color Red
    uint8_t G;           // Color Green
    uint8_t B;           // Color blue
    uint64_t timestamp;  // UTC timestamp
};

struct PointRGBI {
    static constexpr const char* IOX2_TYPE_NAME = "PointRGBI";

    float x;          // X coordinate in meters
    float y;          // Y coordinate in meters
    float z;          // Z coordinate in meters
    uint8_t R;        // Color Red
    uint8_t G;        // Color Green
    uint8_t B;        // Color blue
    float intensity;  // Reflectivity as percentage
};

struct PointRGBIT {
    static constexpr const char* IOX2_TYPE_NAME = "PointRGBIT";

    float x;             // X coordinate in meters
    float y;             // Y coordinate in meters
    float z;             // Z coordinate in meters
    uint8_t R;           // Color Red
    uint8_t G;           // Color Green
    uint8_t B;           // Color blue
    float intensity;     // Reflectivity as percentage
    uint64_t timestamp;  // UTC timestamp
};

// 点云数据结构
// Point cloud data structure with C++20 enhancements
template <typename PointType>
struct PointCloud {
    static constexpr size_t MAX_POINTS = 100000;

    // Type aliases for better template programming
    using value_type = PointType;
    using iterator = typename iox::vector<PointType, MAX_POINTS>::iterator;
    using const_iterator = typename iox::vector<PointType, MAX_POINTS>::const_iterator;

    static constexpr const char* IOX2_TYPE_NAME = "PointCloud";

    uint64_t timestamp;
    uint32_t seq;
    uint32_t num_points;
    iox::vector<PointType, MAX_POINTS> points;

    // range support
    constexpr auto begin() noexcept { return points.begin(); }
    constexpr auto end() noexcept { return points.end(); }
    constexpr auto begin() const noexcept { return points.cbegin(); }
    constexpr auto end() const noexcept { return points.cend(); }
    constexpr auto cbegin() const noexcept { return points.cbegin(); }
    constexpr auto cend() const noexcept { return points.cend(); }

    // Modern utility functions
    constexpr bool empty() const noexcept { return points.empty(); }
    constexpr size_t size() const noexcept { return points.size(); }
    constexpr void clear() noexcept
    {
        points.clear();
        num_points = 0;
    }

    // Efficient point addition with perfect forwarding
    template <typename... Args>
    constexpr void emplace_point(Args&&... args)
    {
        points.emplace_back(std::forward<Args>(args)...);
        num_points = static_cast<uint32_t>(points.size());
    }
};

template <typename PointType>
inline auto operator<<(std::ostream& stream, const PointCloud<PointType>& cloud) -> std::ostream&
{
    stream << "PointCloud { timestamp: " << cloud.timestamp << ", seq: " << cloud.seq << ", num_points: " << cloud.num_points << " }";
    return stream;
}

// Image buffer format structure
struct ImageFormat {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint8_t pixel_format[4];  // e.g., "RGB8", "BGR8", "GRAY", etc.

    size_t byte_size() const { return stride * height; }
};

inline auto operator<<(std::ostream& stream, const ImageFormat& format) -> std::ostream&
{
    stream << "ImageFormat { width: " << format.width << ", height: " << format.height << ", stride: " << format.stride
           << ", format: " << std::string(reinterpret_cast<const char*>(format.pixel_format), 4) << " }";
    return stream;
}

// Image data structure
struct Image {
    static constexpr const char* IOX2_TYPE_NAME = "Image";
    static constexpr size_t MAX_IMAGE_SIZE = 1920 * 1080 * 3;  // Max size for HD RGB image

    uint64_t timestamp;                         // Timestamp when the image was captured
    uint32_t seq;                               // Sequence number
    ImageFormat format;                         // Image format information
    iox::vector<uint8_t, MAX_IMAGE_SIZE> data;  // Image pixel data
};

inline auto operator<<(std::ostream& stream, const Image& image) -> std::ostream&
{
    stream << "Image { timestamp: " << image.timestamp << ", seq: " << image.seq << ", format: " << image.format
           << ", data_size: " << image.data.size() << " }";
    return stream;
}

}  // namespace ms_slam::slam_common