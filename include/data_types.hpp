#pragma once

#include <cstdint>
#include <iostream>

#include <iox2/iceoryx2.hpp>

// Point cloud data structure compatible with Copper's PointCloud
struct PointCloudPoint {
    static constexpr const char* IOX2_TYPE_NAME = "PointCloudPoint";

    uint64_t tov;        // Time of Validity (nanoseconds since epoch)
    float x;             // X coordinate in meters
    float y;             // Y coordinate in meters
    float z;             // Z coordinate in meters
    float intensity;     // Reflectivity as percentage
    uint8_t return_order; // 0 for first return, 1 for second return, etc.
};

inline auto operator<<(std::ostream& stream, const PointCloudPoint& point) -> std::ostream& {
    stream << "Point { tov: " << point.tov
           << ", x: " << point.x << ", y: " << point.y << ", z: " << point.z
           << ", intensity: " << point.intensity
           << ", return_order: " << static_cast<int>(point.return_order) << " }";
    return stream;
}

// Point cloud data structure
struct PointCloud {
    static constexpr const char* IOX2_TYPE_NAME = "PointCloud";
    static constexpr size_t MAX_POINTS = 100000; // Maximum number of points in a cloud

    uint64_t timestamp;    // Timestamp when the point cloud was captured
    uint32_t seq;          // Sequence number
    uint32_t num_points;   // Number of valid points in the cloud
    iox::vector<PointCloudPoint, MAX_POINTS> points; // Point data
};

inline auto operator<<(std::ostream& stream, const PointCloud& cloud) -> std::ostream& {
    stream << "PointCloud { timestamp: " << cloud.timestamp
           << ", seq: " << cloud.seq
           << ", num_points: " << cloud.num_points << " }";
    return stream;
}

// Image buffer format structure
struct ImageFormat {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint8_t pixel_format[4]; // e.g., "RGB8", "BGR8", "GRAY", etc.

    size_t byte_size() const {
        return stride * height;
    }
};

inline auto operator<<(std::ostream& stream, const ImageFormat& format) -> std::ostream& {
    stream << "ImageFormat { width: " << format.width
           << ", height: " << format.height
           << ", stride: " << format.stride
           << ", format: " << std::string(reinterpret_cast<const char*>(format.pixel_format), 4) << " }";
    return stream;
}

// Image data structure
struct Image {
    static constexpr const char* IOX2_TYPE_NAME = "Image";
    static constexpr size_t MAX_IMAGE_SIZE = 1920 * 1080 * 3; // Max size for HD RGB image

    uint64_t timestamp;    // Timestamp when the image was captured
    uint32_t seq;          // Sequence number
    ImageFormat format;    // Image format information
    iox::vector<uint8_t, MAX_IMAGE_SIZE> data; // Image pixel data
};

inline auto operator<<(std::ostream& stream, const Image& image) -> std::ostream& {
    stream << "Image { timestamp: " << image.timestamp
           << ", seq: " << image.seq
           << ", format: " << image.format
           << ", data_size: " << image.data.size() << " }";
    return stream;
}
