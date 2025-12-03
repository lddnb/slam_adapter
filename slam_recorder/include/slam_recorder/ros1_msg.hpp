/**
 * @file ros1_msg.hpp
 * @brief ROS1 消息解析辅助数据结构
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace ros1_detail
{

/**
 * @brief 读取ROS1消息中的基础数值类型
 * @tparam T 待读取的数值类型
 * @param data 当前数据指针
 * @param remaining 剩余可读字节数
 * @param value 输出的解析结果
 * @return 成功读取返回true
 */
template <typename T>
inline bool ReadPrimitive(const uint8_t*& data, size_t& remaining, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>, "ros1 primitive must be trivially copyable");
    if (remaining < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return true;
}

/**
 * @brief 解析ROS1字符串字段
 * @param data 当前数据指针
 * @param remaining 剩余可读字节数
 * @param out 输出字符串
 * @return 成功解析返回true
 */
inline bool ReadString(const uint8_t*& data, size_t& remaining, std::string& out)
{
    uint32_t length = 0;
    if (!ReadPrimitive(data, remaining, length) || remaining < length) {
        return false;
    }
    out.assign(reinterpret_cast<const char*>(data), length);
    data += length;
    remaining -= length;
    return true;
}

/**
 * @brief 解析定长数组字段
 * @tparam T 数组元素类型
 * @tparam N 数组长度
 * @param data 当前数据指针
 * @param remaining 剩余可读字节数
 * @param out 输出数组
 * @return 成功解析返回true
 */
template <typename T, size_t N>
inline bool ReadArray(const uint8_t*& data, size_t& remaining, std::array<T, N>& out)
{
    for (size_t i = 0; i < N; ++i) {
        if (!ReadPrimitive(data, remaining, out[i])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 解析动态序列字段
 * @tparam T 序列元素类型
 * @param data 当前数据指针
 * @param remaining 剩余可读字节数
 * @param out 输出序列
 * @return 成功解析返回true
 */
template <typename T>
inline bool ReadSequence(const uint8_t*& data, size_t& remaining, std::vector<T>& out)
{
    uint32_t count = 0;
    if (!ReadPrimitive(data, remaining, count)) {
        return false;
    }

    out.resize(count);
    for (auto& element : out) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (!ReadPrimitive(data, remaining, element)) {
                return false;
            }
        } else {
            if (!element.Parse(data, remaining)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace ros1_detail

/**
 * @brief ROS1消息头定义
 */
struct ROS1Header {
    uint32_t seq{};         ///< 序列号
    uint32_t stamp_sec{};   ///< 时间戳秒
    uint32_t stamp_nsec{};  ///< 时间戳纳秒
    std::string frame_id;   ///< 坐标系ID

    /**
     * @brief 解析头部信息
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::ReadPrimitive(data, remaining, seq) && ros1_detail::ReadPrimitive(data, remaining, stamp_sec) &&
               ros1_detail::ReadPrimitive(data, remaining, stamp_nsec) && ros1_detail::ReadString(data, remaining, frame_id);
    }
};

/**
 * @brief PointField字段描述
 */
struct ROS1PointField {
    std::string name;    ///< 字段名称
    uint32_t offset{};   ///< 字节偏移
    uint8_t datatype{};  ///< 数据类型编码
    uint32_t count{};    ///< 元素数量
};

/**
 * @brief ROS1 PointCloud2消息
 */
struct ROS1PointCloud2 {
    ROS1Header header;                   ///< 消息头
    uint32_t height{};                   ///< 点云高度
    uint32_t width{};                    ///< 点云宽度
    std::vector<ROS1PointField> fields;  ///< 字段数组
    bool is_bigendian{};                 ///< 是否大端
    uint32_t point_step{};               ///< 单点字节步长
    uint32_t row_step{};                 ///< 行步长
    std::vector<uint8_t> data;           ///< 点云数据
    bool is_dense{};                     ///< 数据是否密集

    /**
     * @brief 解析PointCloud2消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadPrimitive(data_ptr, remaining, height) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, width)) {
            return false;
        }

        uint32_t field_count = 0;
        if (!ros1_detail::ReadPrimitive(data_ptr, remaining, field_count)) {
            return false;
        }
        fields.resize(field_count);
        for (auto& field : fields) {
            if (!ros1_detail::ReadString(data_ptr, remaining, field.name) || !ros1_detail::ReadPrimitive(data_ptr, remaining, field.offset) ||
                !ros1_detail::ReadPrimitive(data_ptr, remaining, field.datatype) || !ros1_detail::ReadPrimitive(data_ptr, remaining, field.count)) {
                return false;
            }
        }

        uint8_t endian_flag = 0;
        if (!ros1_detail::ReadPrimitive(data_ptr, remaining, endian_flag) || !ros1_detail::ReadPrimitive(data_ptr, remaining, point_step) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, row_step)) {
            return false;
        }
        is_bigendian = endian_flag != 0;

        uint32_t data_length = 0;
        if (!ros1_detail::ReadPrimitive(data_ptr, remaining, data_length) || remaining < data_length) {
            return false;
        }
        data.assign(data_ptr, data_ptr + data_length);
        data_ptr += data_length;
        remaining -= data_length;

        uint8_t dense_flag = 0;
        if (!ros1_detail::ReadPrimitive(data_ptr, remaining, dense_flag)) {
            return false;
        }
        is_dense = dense_flag != 0;

        return remaining == 0;
    }
};

/**
 * @brief ROS1压缩图像消息
 */
struct ROS1CompressedImage {
    ROS1Header header;          ///< 消息头
    std::string format;         ///< 图像格式
    std::vector<uint8_t> data;  ///< 压缩数据

    /**
     * @brief 解析压缩图像
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        uint32_t data_length = 0;
        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadString(data_ptr, remaining, format) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, data_length) || remaining < data_length) {
            return false;
        }

        data.assign(data_ptr, data_ptr + data_length);
        data_ptr += data_length;
        remaining -= data_length;

        return remaining == 0;
    }
};

/**
 * @brief ROS1 原始图像消息
 */
struct ROS1Image {
    ROS1Header header;          ///< 消息头
    uint32_t height{0};         ///< 图像高度
    uint32_t width{0};          ///< 图像宽度
    std::string encoding;       ///< 图像编码（bgr8/rgb8 等）
    uint8_t is_bigendian{0};    ///< 大端标志
    uint32_t step{0};           ///< 行跨度
    std::vector<uint8_t> data;  ///< 原始数据

    /**
     * @brief 解析原始图像消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回 true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        uint32_t data_length = 0;
        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadPrimitive(data_ptr, remaining, height) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, width) || !ros1_detail::ReadString(data_ptr, remaining, encoding) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, is_bigendian) || !ros1_detail::ReadPrimitive(data_ptr, remaining, step) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, data_length) || remaining < data_length) {
            return false;
        }

        data.assign(data_ptr, data_ptr + data_length);
        data_ptr += data_length;
        remaining -= data_length;

        return remaining == 0;
    }
};

/**
 * @brief 三维向量
 */
struct ROS1Vector3 {
    double x{};  ///< x分量
    double y{};  ///< y分量
    double z{};  ///< z分量

    /**
     * @brief 解析向量
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::ReadPrimitive(data, remaining, x) && ros1_detail::ReadPrimitive(data, remaining, y) &&
               ros1_detail::ReadPrimitive(data, remaining, z);
    }
};

/**
 * @brief 四元数
 */
struct ROS1Quaternion {
    double x{};  ///< x分量
    double y{};  ///< y分量
    double z{};  ///< z分量
    double w{};  ///< w分量

    /**
     * @brief 解析四元数
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::ReadPrimitive(data, remaining, x) && ros1_detail::ReadPrimitive(data, remaining, y) &&
               ros1_detail::ReadPrimitive(data, remaining, z) && ros1_detail::ReadPrimitive(data, remaining, w);
    }
};

/**
 * @brief 位姿信息
 */
struct ROS1Pose {
    ROS1Vector3 position;        ///< 位置
    ROS1Quaternion orientation;  ///< 姿态

    /**
     * @brief 解析位姿
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining) { return position.Parse(data, remaining) && orientation.Parse(data, remaining); }
};

/**
 * @brief 带协方差的位姿
 */
struct ROS1PoseWithCovariance {
    ROS1Pose pose;                        ///< 位姿
    std::array<double, 36> covariance{};  ///< 协方差矩阵

    /**
     * @brief 解析带协方差位姿
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining) { return pose.Parse(data, remaining) && ros1_detail::ReadArray(data, remaining, covariance); }
};

/**
 * @brief 速度信息
 */
struct ROS1Twist {
    ROS1Vector3 linear;   ///< 线速度
    ROS1Vector3 angular;  ///< 角速度

    /**
     * @brief 解析速度
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining) { return linear.Parse(data, remaining) && angular.Parse(data, remaining); }
};

/**
 * @brief 带协方差的速度信息
 */
struct ROS1TwistWithCovariance {
    ROS1Twist twist;                      ///< 速度
    std::array<double, 36> covariance{};  ///< 协方差矩阵

    /**
     * @brief 解析带协方差速度
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return twist.Parse(data, remaining) && ros1_detail::ReadArray(data, remaining, covariance);
    }
};

/**
 * @brief 带时间戳的位姿
 */
struct ROS1PoseStamped {
    ROS1Header header;  ///< 消息头
    ROS1Pose pose;      ///< 位姿

    /**
     * @brief 解析带时间戳位姿
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining) { return header.Parse(data, remaining) && pose.Parse(data, remaining); }
};

/**
 * @brief 坐标变换
 */
struct ROS1Transform {
    ROS1Vector3 translation;  ///< 平移
    ROS1Quaternion rotation;  ///< 旋转

    /**
     * @brief 解析坐标变换
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining) { return translation.Parse(data, remaining) && rotation.Parse(data, remaining); }
};

/**
 * @brief 带时间戳的坐标变换
 */
struct ROS1TransformStamped {
    ROS1Header header;           ///< 消息头
    std::string child_frame_id;  ///< 子坐标系ID
    ROS1Transform transform;     ///< 坐标变换

    /**
     * @brief 解析带时间戳的坐标变换
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return header.Parse(data, remaining) && ros1_detail::ReadString(data, remaining, child_frame_id) && transform.Parse(data, remaining);
    }
};

/**
 * @brief TF消息容器
 */
struct ROS1TFMessage {
    std::vector<ROS1TransformStamped> transforms;  ///< 变换数组

    /**
     * @brief 解析TF消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        return ros1_detail::ReadSequence(data_ptr, remaining, transforms) && remaining == 0;
    }
};

/**
 * @brief ROS1轨迹消息
 */
struct ROS1Path {
    ROS1Header header;                   ///< 消息头
    std::vector<ROS1PoseStamped> poses;  ///< 位姿序列

    /**
     * @brief 解析轨迹消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadSequence(data_ptr, remaining, poses)) {
            return false;
        }
        return remaining == 0;
    }
};

/**
 * @brief Livox定制点结构
 */
struct ROS1LivoxCustomPoint {
    uint32_t offset_time{};  ///< 相对于帧首的时间偏移
    float x{};               ///< x坐标
    float y{};               ///< y坐标
    float z{};               ///< z坐标
    uint8_t reflectivity{};  ///< 反射率
    uint8_t tag{};           ///< 点标签
    uint8_t line{};          ///< 激光线号

    /**
     * @brief 解析Livox点
     * @param data 当前数据指针
     * @param remaining 剩余可读字节数
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::ReadPrimitive(data, remaining, offset_time) && ros1_detail::ReadPrimitive(data, remaining, x) &&
               ros1_detail::ReadPrimitive(data, remaining, y) && ros1_detail::ReadPrimitive(data, remaining, z) &&
               ros1_detail::ReadPrimitive(data, remaining, reflectivity) && ros1_detail::ReadPrimitive(data, remaining, tag) &&
               ros1_detail::ReadPrimitive(data, remaining, line);
    }
};

/**
 * @brief Livox定制消息
 */
struct ROS1LivoxCustomMsg {
    ROS1Header header;                         ///< 消息头
    uint64_t timebase{};                       ///< 时间基准
    uint32_t point_num{};                      ///< 点数量
    uint8_t lidar_id{};                        ///< 雷达ID
    std::array<uint8_t, 3> reserved{};         ///< 预留字段
    std::vector<ROS1LivoxCustomPoint> points;  ///< 点云数据

    /**
     * @brief 解析Livox定制消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        uint32_t serialized_point_count = 0;
        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadPrimitive(data_ptr, remaining, timebase) ||
            !ros1_detail::ReadPrimitive(data_ptr, remaining, point_num) || !ros1_detail::ReadPrimitive(data_ptr, remaining, lidar_id) ||
            !ros1_detail::ReadArray(data_ptr, remaining, reserved) || !ros1_detail::ReadPrimitive(data_ptr, remaining, serialized_point_count)) {
            return false;
        }

        points.resize(serialized_point_count);
        for (auto& point : points) {
            if (!point.Parse(data_ptr, remaining)) {
                return false;
            }
        }

        if (point_num != serialized_point_count) {
            point_num = serialized_point_count;
        }
        return remaining == 0;
    }
};

/**
 * @brief ROS1 IMU消息
 */
struct ROS1Imu {
    ROS1Header header;                                       ///< 消息头
    ROS1Quaternion orientation;                              ///< 姿态
    std::array<double, 9> orientation_covariance{};          ///< 姿态协方差
    ROS1Vector3 angular_velocity;                            ///< 角速度
    std::array<double, 9> angular_velocity_covariance{};     ///< 角速度协方差
    ROS1Vector3 linear_acceleration;                         ///< 线加速度
    std::array<double, 9> linear_acceleration_covariance{};  ///< 线加速度协方差

    /**
     * @brief 解析IMU消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        return header.Parse(data_ptr, remaining) && orientation.Parse(data_ptr, remaining) &&
               ros1_detail::ReadArray(data_ptr, remaining, orientation_covariance) && angular_velocity.Parse(data_ptr, remaining) &&
               ros1_detail::ReadArray(data_ptr, remaining, angular_velocity_covariance) && linear_acceleration.Parse(data_ptr, remaining) &&
               ros1_detail::ReadArray(data_ptr, remaining, linear_acceleration_covariance) && remaining == 0;
    }
};

/**
 * @brief ROS1里程计消息
 */
struct ROS1Odometry {
    ROS1Header header;                          ///< 消息头
    std::string child_frame_id;                 ///< 子坐标系
    ROS1Pose pose;                              ///< 位姿
    std::array<double, 36> pose_covariance{};   ///< 位姿协方差
    ROS1Twist twist;                            ///< 速度
    std::array<double, 36> twist_covariance{};  ///< 速度协方差

    /**
     * @brief 解析里程计消息
     * @param msg_data 原始消息指针
     * @param msg_size 原始消息长度
     * @return 成功解析返回true
     */
    bool Parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        ROS1PoseWithCovariance pose_with_cov;
        ROS1TwistWithCovariance twist_with_cov;

        if (!header.Parse(data_ptr, remaining) || !ros1_detail::ReadString(data_ptr, remaining, child_frame_id) ||
            !pose_with_cov.Parse(data_ptr, remaining) || !twist_with_cov.Parse(data_ptr, remaining)) {
            return false;
        }

        pose = pose_with_cov.pose;
        pose_covariance = pose_with_cov.covariance;
        twist = twist_with_cov.twist;
        twist_covariance = twist_with_cov.covariance;

        return remaining == 0;
    }
};
