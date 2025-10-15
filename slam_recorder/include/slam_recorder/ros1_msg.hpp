/**
 * @file ros1_msg.hpp
 * @brief ROS1 消息解析辅助数据结构
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief ROS1 Header 结构体
 */
struct ROS1Header {
    uint32_t seq{};        ///< 序列号
    uint32_t stamp_sec{};  ///< 时间戳秒
    uint32_t stamp_nsec{}; ///< 时间戳纳秒
    std::string frame_id;  ///< 坐标系名称

    /**
     * @brief 从字节流解析 Header
     * @param data 指向数据指针，会在解析后前移
     * @param remaining 剩余字节数，会在解析后递减
     * @return 解析成功返回 true
     */
    bool parse(const uint8_t*& data, size_t& remaining);
};

/**
 * @brief ROS1 PointField 定义
 */
struct ROS1PointField {
    std::string name;  ///< 字段名称
    uint32_t offset{}; ///< 偏移
    uint8_t datatype{}; ///< 数据类型
    uint32_t count{};  ///< 元素数量
};

/**
 * @brief ROS1 PointCloud2 消息体
 */
struct ROS1PointCloud2 {
    ROS1Header header;                     ///< 消息头
    uint32_t height{};                     ///< 点云高度
    uint32_t width{};                      ///< 点云宽度
    std::vector<ROS1PointField> fields;    ///< 字段数组
    bool is_bigendian{};                   ///< 大端标记
    uint32_t point_step{};                 ///< 单点字节数
    uint32_t row_step{};                   ///< 每行字节数
    std::vector<uint8_t> data;             ///< 数据区
    bool is_dense{};                       ///< 是否稠密

    /**
     * @brief 解析 PointCloud2 消息
     * @param msg_data 消息原始字节
     * @param msg_size 字节长度
     * @return 解析成功返回 true
     */
    bool parse(const uint8_t* msg_data, size_t msg_size);
};

/**
 * @brief ROS1 CompressedImage 消息体
 */
struct ROS1CompressedImage {
    ROS1Header header;             ///< 消息头
    std::string format;            ///< 图像格式
    std::vector<uint8_t> data;     ///< 压缩数据

    /**
     * @brief 解析 CompressedImage 消息
     * @param msg_data 消息原始字节
     * @param msg_size 字节长度
     * @return 解析成功返回 true
     */
    bool parse(const uint8_t* msg_data, size_t msg_size);
};

/**
 * @brief ROS1 Imu 消息体（简化版）
 */
struct ROS1Imu {
    ROS1Header header; ///< 消息头

    struct Quaternion {
        double x{};
        double y{};
        double z{};
        double w{};
    } orientation; ///< 姿态四元数

    double orientation_covariance[9]{}; ///< 姿态协方差

    struct Vector3 {
        double x{};
        double y{};
        double z{};
    } angular_velocity; ///< 角速度

    double angular_velocity_covariance[9]{}; ///< 角速度协方差

    struct Vector3 linear_acceleration; ///< 线加速度

    double linear_acceleration_covariance[9]{}; ///< 线加速度协方差

    /**
     * @brief 解析 Imu 消息
     * @param msg_data 消息原始字节
     * @param msg_size 字节长度
     * @return 解析成功返回 true
     */
    bool parse(const uint8_t* msg_data, size_t msg_size);
};

inline bool ROS1Header::parse(const uint8_t*& data, size_t& remaining)
{
    if (remaining < 12) return false;

    seq = *reinterpret_cast<const uint32_t*>(data);
    data += 4;
    remaining -= 4;

    stamp_sec = *reinterpret_cast<const uint32_t*>(data);
    data += 4;
    remaining -= 4;

    stamp_nsec = *reinterpret_cast<const uint32_t*>(data);
    data += 4;
    remaining -= 4;

    if (remaining < 4) return false;
    uint32_t str_len = *reinterpret_cast<const uint32_t*>(data);
    data += 4;
    remaining -= 4;

    if (remaining < str_len) return false;
    frame_id = std::string(reinterpret_cast<const char*>(data), str_len);
    data += str_len;
    remaining -= str_len;

    return true;
}

inline bool ROS1PointCloud2::parse(const uint8_t* msg_data, size_t msg_size)
{
    const uint8_t* data_ptr = msg_data;
    size_t remaining = msg_size;

    if (!header.parse(data_ptr, remaining)) return false;

    if (remaining < 8) return false;
    height = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    width = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    if (remaining < 4) return false;
    uint32_t fields_len = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    fields.resize(fields_len);
    for (uint32_t i = 0; i < fields_len; ++i) {
        if (remaining < 4) return false;
        uint32_t name_len = *reinterpret_cast<const uint32_t*>(data_ptr);
        data_ptr += 4;
        remaining -= 4;

        if (remaining < name_len) return false;
        fields[i].name = std::string(reinterpret_cast<const char*>(data_ptr), name_len);
        data_ptr += name_len;
        remaining -= name_len;

        if (remaining < 9) return false;
        fields[i].offset = *reinterpret_cast<const uint32_t*>(data_ptr);
        data_ptr += 4;
        remaining -= 4;

        fields[i].datatype = *data_ptr;
        data_ptr += 1;
        remaining -= 1;

        fields[i].count = *reinterpret_cast<const uint32_t*>(data_ptr);
        data_ptr += 4;
        remaining -= 4;
    }

    if (remaining < 9) return false;
    is_bigendian = *data_ptr != 0;
    data_ptr += 1;
    remaining -= 1;

    point_step = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    row_step = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    if (remaining < 4) return false;
    uint32_t data_len = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    if (remaining < data_len) return false;
    data.assign(data_ptr, data_ptr + data_len);
    data_ptr += data_len;
    remaining -= data_len;

    if (remaining < 1) return false;
    is_dense = *data_ptr != 0;

    return true;
}

inline bool ROS1CompressedImage::parse(const uint8_t* msg_data, size_t msg_size)
{
    const uint8_t* data_ptr = msg_data;
    size_t remaining = msg_size;

    if (!header.parse(data_ptr, remaining)) return false;

    if (remaining < 4) return false;
    uint32_t format_len = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    if (remaining < format_len) return false;
    format = std::string(reinterpret_cast<const char*>(data_ptr), format_len);
    data_ptr += format_len;
    remaining -= format_len;

    if (remaining < 4) return false;
    uint32_t data_len = *reinterpret_cast<const uint32_t*>(data_ptr);
    data_ptr += 4;
    remaining -= 4;

    if (remaining < data_len) return false;
    data.assign(data_ptr, data_ptr + data_len);

    return true;
}

inline bool ROS1Imu::parse(const uint8_t* msg_data, size_t msg_size)
{
    const uint8_t* data_ptr = msg_data;
    size_t remaining = msg_size;

    if (!header.parse(data_ptr, remaining)) return false;

    if (remaining < 32) return false;
    orientation.x = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    orientation.y = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    orientation.z = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    orientation.w = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;

    if (remaining < 72) return false;
    for (int i = 0; i < 9; ++i) {
        orientation_covariance[i] = *reinterpret_cast<const double*>(data_ptr);
        data_ptr += 8;
        remaining -= 8;
    }

    if (remaining < 24) return false;
    angular_velocity.x = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    angular_velocity.y = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    angular_velocity.z = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;

    if (remaining < 72) return false;
    for (int i = 0; i < 9; ++i) {
        angular_velocity_covariance[i] = *reinterpret_cast<const double*>(data_ptr);
        data_ptr += 8;
        remaining -= 8;
    }

    if (remaining < 24) return false;
    linear_acceleration.x = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    linear_acceleration.y = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;
    linear_acceleration.z = *reinterpret_cast<const double*>(data_ptr);
    data_ptr += 8;
    remaining -= 8;

    if (remaining < 72) return false;
    for (int i = 0; i < 9; ++i) {
        linear_acceleration_covariance[i] = *reinterpret_cast<const double*>(data_ptr);
        data_ptr += 8;
        remaining -= 8;
    }

    return true;
}

