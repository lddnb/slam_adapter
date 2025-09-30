#pragma once

#include "slam_common/iceoryx_data_types.hpp"
#include "fbs/slam_data_generated.h"  // FlatBuffers 生成的头文件

#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <cstring>

namespace ms_slam::slam_common
{

// ============ Image 序列化方法实现 ============

inline flatbuffers::DetachedBuffer Image::serialize() const
{
    flatbuffers::FlatBufferBuilder builder(this->byte_size() + 1024);

    // 创建 ImageFormat
    auto format_msg = SlamData::CreateImageFormat(
        builder,
        this->format.width,
        this->format.height,
        this->format.stride,
        static_cast<SlamData::PixelFormat>(this->format.pixel_format));

    // 创建图像数据 vector
    auto data_vec = builder.CreateVector(this->data.data(), this->data.size());

    // 创建 ImageMsg
    auto image_msg = SlamData::CreateImageMsg(
        builder,
        this->timestamp,
        this->seq,
        format_msg,
        data_vec);

    builder.Finish(image_msg);
    return builder.Release();
}

inline Image Image::deserialize(const uint8_t* buffer, size_t size,
                                std::pmr::memory_resource* mr)
{
    auto msg = flatbuffers::GetRoot<SlamData::ImageMsg>(buffer);

    Image img(mr);
    img.timestamp = msg->timestamp();
    img.seq = msg->seq();

    // 解析 format
    auto format = msg->format();
    img.format.width = format->width();
    img.format.height = format->height();
    img.format.stride = format->stride();
    img.format.pixel_format = static_cast<pixel_format_t>(format->pixel_format());

    // 零拷贝访问数据（只在读取时零拷贝，如需修改仍需复制）
    auto data = msg->data();
    img.data.assign(data->begin(), data->end());

    return img;
}

// ============ 保留原有的 ImageSerializer（向后兼容）============
class ImageSerializer
{
  public:
    // 序列化 Image 到 FlatBuffers
    static flatbuffers::DetachedBuffer serialize(const Image& img)
    {
        return img.serialize();
    }

    // 反序列化 FlatBuffers 到 Image
    static Image deserialize(const uint8_t* buffer, size_t size,
                            std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    {
        return Image::deserialize(buffer, size, mr);
    }
};

// ============ PointCloud 序列化方法实现（模板特化）============

// PointCloudIT 序列化实现
template <>
inline flatbuffers::DetachedBuffer PointCloudSoA<PointIT, std::pmr::polymorphic_allocator<float>>::serialize() const
{
    flatbuffers::FlatBufferBuilder builder(this->num_points * 32 + 1024);

    auto x_vec = builder.CreateVector(this->x.data(), this->x.size());
    auto y_vec = builder.CreateVector(this->y.data(), this->y.size());
    auto z_vec = builder.CreateVector(this->z.data(), this->z.size());
    auto intensity_vec = builder.CreateVector(this->intensity.data(), this->intensity.size());
    auto timestamps_vec = builder.CreateVector(this->timestamps.data(), this->timestamps.size());

    auto msg = SlamData::CreatePointCloudITMsg(
        builder,
        this->timestamp,
        this->seq,
        this->num_points,
        x_vec, y_vec, z_vec,
        intensity_vec,
        timestamps_vec);

    builder.Finish(msg);
    return builder.Release();
}

template <>
inline PointCloudSoA<PointIT, std::pmr::polymorphic_allocator<float>>
PointCloudSoA<PointIT, std::pmr::polymorphic_allocator<float>>::deserialize(
    const uint8_t* buffer, size_t size,
    std::pmr::memory_resource* mr)
{
    auto msg = flatbuffers::GetRoot<SlamData::PointCloudITMsg>(buffer);

    PointCloudSoA<PointIT, std::pmr::polymorphic_allocator<float>> cloud{std::pmr::polymorphic_allocator<float>(mr)};
    cloud.timestamp = msg->timestamp();
    cloud.seq = msg->seq();
    cloud.num_points = msg->num_points();

    cloud.x.assign(msg->x()->begin(), msg->x()->end());
    cloud.y.assign(msg->y()->begin(), msg->y()->end());
    cloud.z.assign(msg->z()->begin(), msg->z()->end());
    cloud.intensity.assign(msg->intensity()->begin(), msg->intensity()->end());
    cloud.timestamps.assign(msg->timestamps()->begin(), msg->timestamps()->end());

    return cloud;
}

// PointCloudRGB 序列化实现
template <>
inline flatbuffers::DetachedBuffer PointCloudSoA<PointRGB, std::pmr::polymorphic_allocator<float>>::serialize() const
{
    flatbuffers::FlatBufferBuilder builder(this->num_points * 16 + 1024);

    auto x_vec = builder.CreateVector(this->x.data(), this->x.size());
    auto y_vec = builder.CreateVector(this->y.data(), this->y.size());
    auto z_vec = builder.CreateVector(this->z.data(), this->z.size());
    auto r_vec = builder.CreateVector(this->r.data(), this->r.size());
    auto g_vec = builder.CreateVector(this->g.data(), this->g.size());
    auto b_vec = builder.CreateVector(this->b.data(), this->b.size());

    auto msg = SlamData::CreatePointCloudRGBMsg(
        builder,
        this->timestamp,
        this->seq,
        this->num_points,
        x_vec, y_vec, z_vec,
        r_vec, g_vec, b_vec);

    builder.Finish(msg);
    return builder.Release();
}

template <>
inline PointCloudSoA<PointRGB, std::pmr::polymorphic_allocator<float>>
PointCloudSoA<PointRGB, std::pmr::polymorphic_allocator<float>>::deserialize(
    const uint8_t* buffer, size_t size,
    std::pmr::memory_resource* mr)
{
    auto msg = flatbuffers::GetRoot<SlamData::PointCloudRGBMsg>(buffer);

    PointCloudSoA<PointRGB, std::pmr::polymorphic_allocator<float>> cloud{std::pmr::polymorphic_allocator<float>(mr)};
    cloud.timestamp = msg->timestamp();
    cloud.seq = msg->seq();
    cloud.num_points = msg->num_points();

    cloud.x.assign(msg->x()->begin(), msg->x()->end());
    cloud.y.assign(msg->y()->begin(), msg->y()->end());
    cloud.z.assign(msg->z()->begin(), msg->z()->end());
    cloud.r.assign(msg->r()->begin(), msg->r()->end());
    cloud.g.assign(msg->g()->begin(), msg->g()->end());
    cloud.b.assign(msg->b()->begin(), msg->b()->end());

    return cloud;
}

// PointCloudRGBIT 序列化实现
template <>
inline flatbuffers::DetachedBuffer PointCloudSoA<PointRGBIT, std::pmr::polymorphic_allocator<float>>::serialize() const
{
    flatbuffers::FlatBufferBuilder builder(this->num_points * 32 + 1024);

    auto x_vec = builder.CreateVector(this->x.data(), this->x.size());
    auto y_vec = builder.CreateVector(this->y.data(), this->y.size());
    auto z_vec = builder.CreateVector(this->z.data(), this->z.size());
    auto r_vec = builder.CreateVector(this->r.data(), this->r.size());
    auto g_vec = builder.CreateVector(this->g.data(), this->g.size());
    auto b_vec = builder.CreateVector(this->b.data(), this->b.size());
    auto intensity_vec = builder.CreateVector(this->intensity.data(), this->intensity.size());
    auto timestamps_vec = builder.CreateVector(this->timestamps.data(), this->timestamps.size());

    auto msg = SlamData::CreatePointCloudRGBITMsg(
        builder,
        this->timestamp,
        this->seq,
        this->num_points,
        x_vec, y_vec, z_vec,
        r_vec, g_vec, b_vec,
        intensity_vec,
        timestamps_vec);

    builder.Finish(msg);
    return builder.Release();
}

template <>
inline PointCloudSoA<PointRGBIT, std::pmr::polymorphic_allocator<float>>
PointCloudSoA<PointRGBIT, std::pmr::polymorphic_allocator<float>>::deserialize(
    const uint8_t* buffer, size_t size,
    std::pmr::memory_resource* mr)
{
    auto msg = flatbuffers::GetRoot<SlamData::PointCloudRGBITMsg>(buffer);

    PointCloudSoA<PointRGBIT, std::pmr::polymorphic_allocator<float>> cloud{std::pmr::polymorphic_allocator<float>(mr)};
    cloud.timestamp = msg->timestamp();
    cloud.seq = msg->seq();
    cloud.num_points = msg->num_points();

    cloud.x.assign(msg->x()->begin(), msg->x()->end());
    cloud.y.assign(msg->y()->begin(), msg->y()->end());
    cloud.z.assign(msg->z()->begin(), msg->z()->end());
    cloud.r.assign(msg->r()->begin(), msg->r()->end());
    cloud.g.assign(msg->g()->begin(), msg->g()->end());
    cloud.b.assign(msg->b()->begin(), msg->b()->end());
    cloud.intensity.assign(msg->intensity()->begin(), msg->intensity()->end());
    cloud.timestamps.assign(msg->timestamps()->begin(), msg->timestamps()->end());

    return cloud;
}

// ============ PointCloud 序列化器（模板，向后兼容）============
template <typename PointType>
class PointCloudSerializer;

// PointIT 特化
template <>
class PointCloudSerializer<PointIT>
{
  public:
    using PointCloudType = PointCloudSoA<PointIT>;

    static flatbuffers::DetachedBuffer serialize(const PointCloudType& cloud)
    {
        return cloud.serialize();
    }

    static PointCloudType deserialize(const uint8_t* buffer, size_t size,
                                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    {
        return PointCloudType::deserialize(buffer, size, mr);
    }
};

// PointRGB 特化
template <>
class PointCloudSerializer<PointRGB>
{
  public:
    using PointCloudType = PointCloudSoA<PointRGB>;

    static flatbuffers::DetachedBuffer serialize(const PointCloudType& cloud)
    {
        return cloud.serialize();
    }

    static PointCloudType deserialize(const uint8_t* buffer, size_t size,
                                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    {
        return PointCloudType::deserialize(buffer, size, mr);
    }
};

// PointRGBIT 特化
template <>
class PointCloudSerializer<PointRGBIT>
{
  public:
    using PointCloudType = PointCloudSoA<PointRGBIT>;

    static flatbuffers::DetachedBuffer serialize(const PointCloudType& cloud)
    {
        return cloud.serialize();
    }

    static PointCloudType deserialize(const uint8_t* buffer, size_t size,
                                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    {
        return PointCloudType::deserialize(buffer, size, mr);
    }
};

}  // namespace ms_slam::slam_common