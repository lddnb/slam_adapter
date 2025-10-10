#pragma once

#include <flatbuffers/flatbuffers.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>

namespace ms_slam::slam_common
{

// ============================================================================
// Foxglove FlatBuffers 消息包装器
// ============================================================================
//
// 这些包装器类型为 Foxglove FlatBuffers 消息提供零拷贝的序列化接口，
// 满足 Serializable 概念的要求，可以直接用于 GenericFlatBufferPublisher
// 和 GenericFlatBufferSubscriber。
//
// 使用方式:
//   1. 订阅端: 使用 deserialize() 从字节流恢复原生 Foxglove 类型
//   2. 发布端: 使用 publish_raw() 或 publish_from_builder() 直接发布
//
// ============================================================================

template <typename Msg>
class FoxgloveMsg
{
  public:
    /// 从已序列化的字节流构造（订阅端使用）
    explicit FoxgloveMsg(const uint8_t* data, size_t size)
        : data_(data, data + size)
    {
        // 验证 FlatBuffers 数据
        flatbuffers::Verifier verifier(data, size);
        if (!verifier.VerifyBuffer<Msg>()) {
            throw std::runtime_error("Invalid PointCloud FlatBuffers data");
        }
    }

    /// 获取原生 Foxglove PointCloud 指针（零拷贝）
    const Msg* get() const
    {
        return flatbuffers::GetRoot<Msg>(data_.data());
    }

    /// 实现 Serializable 概念要求的 serialize() 方法
    flatbuffers::DetachedBuffer serialize() const
    {
        // 创建一个新的 FlatBufferBuilder 并复制数据
        flatbuffers::FlatBufferBuilder builder(data_.size());
        builder.PushBytes(data_.data(), data_.size());
        return builder.Release();
    }

    /// 实现 Serializable 概念要求的 deserialize() 静态方法
    static FoxgloveMsg deserialize(const uint8_t* buffer, size_t size)
    {
        return FoxgloveMsg(buffer, size);
    }

    /// 获取原始数据指针（用于零拷贝发布）
    const uint8_t* raw_data() const { return data_.data(); }
    size_t raw_size() const { return data_.size(); }

  private:
    std::vector<uint8_t> data_;  // 存储原始 FlatBuffers 字节流
};

using FoxglovePointCloud = FoxgloveMsg<foxglove::PointCloud>;
using FoxgloveCompressedImage = FoxgloveMsg<foxglove::CompressedImage>;
using FoxgloveImu = FoxgloveMsg<foxglove::Imu>;

}  // namespace ms_slam::slam_common
