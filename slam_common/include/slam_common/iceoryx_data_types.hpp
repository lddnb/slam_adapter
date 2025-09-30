#pragma once

#include <cstdint>
#include <iostream>
#include <memory_resource>
#include <vector>

#include <iox2/iceoryx2.hpp>

// 前向声明 flatbuffers
namespace flatbuffers {
class DetachedBuffer;
}

namespace ms_slam::slam_common
{
// ============ 点类型定义（保持不变）============
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

// ============ 点云数据结构（模板，SoA布局）============
// 使用 std::pmr 以支持内存池优化
template <typename PointType, typename Allocator = std::pmr::polymorphic_allocator<float>>
struct PointCloudSoA {
    using value_type = PointType;

    static constexpr const char* IOX2_TYPE_NAME = "PointCloudSoA";

    uint64_t timestamp;
    uint32_t seq;
    uint32_t num_points;

    // SoA 布局：所有点的 x, y, z 分别存储
    std::vector<float, Allocator> x;
    std::vector<float, Allocator> y;
    std::vector<float, Allocator> z;

    // 可选属性（根据 PointType 不同而不同）
    std::vector<float, Allocator> intensity;
    std::vector<uint64_t, std::pmr::polymorphic_allocator<uint64_t>> timestamps;
    std::vector<uint8_t, std::pmr::polymorphic_allocator<uint8_t>> r;
    std::vector<uint8_t, std::pmr::polymorphic_allocator<uint8_t>> g;
    std::vector<uint8_t, std::pmr::polymorphic_allocator<uint8_t>> b;

    // 构造函数
    explicit PointCloudSoA(const Allocator& alloc = Allocator())
        : num_points(0), x(alloc), y(alloc), z(alloc), intensity(alloc),
          timestamps(alloc.resource()), r(alloc.resource()), g(alloc.resource()), b(alloc.resource()) {}

    // 预分配
    void reserve(size_t count) {
        x.reserve(count);
        y.reserve(count);
        z.reserve(count);
        if constexpr (requires { PointType::intensity; }) {
            intensity.reserve(count);
        }
        if constexpr (requires { PointType::timestamp; }) {
            timestamps.reserve(count);
        }
        if constexpr (requires { PointType::R; }) {
            r.reserve(count);
            g.reserve(count);
            b.reserve(count);
        }
    }

    // 清空
    void clear() {
        x.clear();
        y.clear();
        z.clear();
        intensity.clear();
        timestamps.clear();
        r.clear();
        g.clear();
        b.clear();
        num_points = 0;
    }

    size_t size() const { return num_points; }
    bool empty() const { return num_points == 0; }

    // 添加点（支持不同点类型的自动适配）
    void emplace_back(const PointType& point) {
        x.push_back(point.x);
        y.push_back(point.y);
        z.push_back(point.z);

        if constexpr (requires { PointType::intensity; }) {
            intensity.push_back(point.intensity);
        }
        if constexpr (requires { PointType::timestamp; }) {
            timestamps.push_back(point.timestamp);
        }
        if constexpr (requires { PointType::R; }) {
            r.push_back(point.R);
            g.push_back(point.G);
            b.push_back(point.B);
        }

        num_points++;
    }

    // 序列化/反序列化方法声明（在 flatbuffer_serializer.hpp 中实现）
    flatbuffers::DetachedBuffer serialize() const;
    static PointCloudSoA deserialize(
        const uint8_t* buffer, size_t size,
        std::pmr::memory_resource* mr = std::pmr::get_default_resource());
};

// 常用点云类型别名
using PointCloudIT = PointCloudSoA<PointIT>;
using PointCloudRGB = PointCloudSoA<PointRGB>;
using PointCloudRGBIT = PointCloudSoA<PointRGBIT>;

// ============ 像素格式枚举 ============
enum pixel_format_t {
    PIXEL_FORMAT_UNKNOWN = 0,
    PIXEL_FORMAT_RGB8 = 1,
    PIXEL_FORMAT_BGR8 = 2,
    PIXEL_FORMAT_GRAY8 = 3,
    PIXEL_FORMAT_BGRA8 = 4
};

// ============ 图像格式 ============
struct ImageFormat {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    pixel_format_t pixel_format;

    size_t byte_size() const { return stride * height; }
};

// ============ 图像数据结构（非模板，使用 std::pmr）============
struct Image {
    static constexpr const char* IOX2_TYPE_NAME = "Image";

    uint64_t timestamp;
    uint32_t seq;
    ImageFormat format;

    // 使用 pmr::vector 支持内存池
    std::pmr::vector<uint8_t> data;

    // 构造函数
    explicit Image(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
        : timestamp(0), seq(0), format{}, data(mr) {}

    Image(uint64_t ts, uint32_t sequence, const ImageFormat& fmt,
          std::pmr::memory_resource* mr = std::pmr::get_default_resource())
        : timestamp(ts), seq(sequence), format(fmt), data(mr) {}

    size_t byte_size() const { return data.size(); }
    uint8_t* data_ptr() { return data.data(); }
    const uint8_t* data_ptr() const { return data.data(); }

    // 序列化/反序列化方法（在 flatbuffer_serializer.hpp 中实现）
    flatbuffers::DetachedBuffer serialize() const;
    static Image deserialize(const uint8_t* buffer, size_t size,
                            std::pmr::memory_resource* mr = std::pmr::get_default_resource());
};

}  // namespace ms_slam::slam_common