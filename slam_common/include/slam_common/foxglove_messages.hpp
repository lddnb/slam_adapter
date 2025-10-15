/**
 * @file foxglove_messages.hpp
 * @brief Foxglove FlatBuffers 消息包装器
 */
#pragma once

#include <stdexcept>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>

namespace ms_slam::slam_common
{
/**
 * @brief Foxglove FlatBuffers 消息零拷贝包装器
 *
 * 使用说明：
 * 1. 订阅端使用 `deserialize()` 从字节流恢复原生 Foxglove 类型。
 * 2. 发布端使用 `publish_raw()` 或 `publish_from_builder()` 进行零拷贝发布。
 */

template <typename Msg>
class FoxgloveMsg
{
  public:
    /**
     * @brief 从已序列化的字节流构造
     * @param data FlatBuffers 数据指针
     * @param size 数据长度（字节）
     */
    explicit FoxgloveMsg(const uint8_t* data, size_t size)
        : data_(data, data + size)
    {
        flatbuffers::Verifier verifier(data, size);
        if (!verifier.VerifyBuffer<Msg>()) {
            throw std::runtime_error("Invalid PointCloud FlatBuffers data");
        }
    }

    /**
     * @brief 获取原生 Foxglove 对象指针
     * @return 原生对象指针（零拷贝）
     */
    const Msg* get() const
    {
        return flatbuffers::GetRoot<Msg>(data_.data());
    }

    /**
     * @brief 序列化为 FlatBuffers 缓冲区
     * @return 可分离缓冲区
     */
    flatbuffers::DetachedBuffer serialize() const
    {
        flatbuffers::FlatBufferBuilder builder(data_.size());
        builder.PushFlatBuffer(data_.data(), data_.size());
        return builder.Release();
    }

    /**
     * @brief 从原始缓冲区反序列化
     * @param buffer 字节流指针
     * @param size 字节长度
     * @return 包装器对象
     */
    static FoxgloveMsg deserialize(const uint8_t* buffer, size_t size)
    {
        return FoxgloveMsg(buffer, size);
    }

    /**
     * @brief 获取原始数据指针
     * @return 原始字节指针
     */
    const uint8_t* raw_data() const { return data_.data(); }

    /**
     * @brief 获取原始数据大小
     * @return 字节长度
     */
    size_t raw_size() const { return data_.size(); }

  private:
    std::vector<uint8_t> data_;  ///< 存储原始 FlatBuffers 字节流
};

using FoxglovePointCloud = FoxgloveMsg<foxglove::PointCloud>;
using FoxgloveCompressedImage = FoxgloveMsg<foxglove::CompressedImage>;
using FoxgloveImu = FoxgloveMsg<foxglove::Imu>;

}  // namespace ms_slam::slam_common
