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

template <typename T>
inline bool read_primitive(const uint8_t*& data, size_t& remaining, T& value)
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

inline bool read_string(const uint8_t*& data, size_t& remaining, std::string& out)
{
    uint32_t length = 0;
    if (!read_primitive(data, remaining, length) || remaining < length) {
        return false;
    }
    out.assign(reinterpret_cast<const char*>(data), length);
    data += length;
    remaining -= length;
    return true;
}

template <typename T, size_t N>
inline bool read_array(const uint8_t*& data, size_t& remaining, std::array<T, N>& out)
{
    for (size_t i = 0; i < N; ++i) {
        if (!read_primitive(data, remaining, out[i])) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline bool read_sequence(const uint8_t*& data, size_t& remaining, std::vector<T>& out)
{
    uint32_t count = 0;
    if (!read_primitive(data, remaining, count)) {
        return false;
    }

    out.resize(count);
    for (auto& element : out) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (!read_primitive(data, remaining, element)) {
                return false;
            }
        } else {
            if (!element.parse(data, remaining)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace ros1_detail

struct ROS1Header {
    uint32_t seq{};
    uint32_t stamp_sec{};
    uint32_t stamp_nsec{};
    std::string frame_id;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::read_primitive(data, remaining, seq) &&
               ros1_detail::read_primitive(data, remaining, stamp_sec) &&
               ros1_detail::read_primitive(data, remaining, stamp_nsec) &&
               ros1_detail::read_string(data, remaining, frame_id);
    }
};

struct ROS1PointField {
    std::string name;
    uint32_t offset{};
    uint8_t datatype{};
    uint32_t count{};
};

struct ROS1PointCloud2 {
    ROS1Header header;
    uint32_t height{};
    uint32_t width{};
    std::vector<ROS1PointField> fields;
    bool is_bigendian{};
    uint32_t point_step{};
    uint32_t row_step{};
    std::vector<uint8_t> data;
    bool is_dense{};

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        if (!header.parse(data_ptr, remaining) ||
            !ros1_detail::read_primitive(data_ptr, remaining, height) ||
            !ros1_detail::read_primitive(data_ptr, remaining, width)) {
            return false;
        }

        uint32_t field_count = 0;
        if (!ros1_detail::read_primitive(data_ptr, remaining, field_count)) {
            return false;
        }
        fields.resize(field_count);
        for (auto& field : fields) {
            if (!ros1_detail::read_string(data_ptr, remaining, field.name) ||
                !ros1_detail::read_primitive(data_ptr, remaining, field.offset) ||
                !ros1_detail::read_primitive(data_ptr, remaining, field.datatype) ||
                !ros1_detail::read_primitive(data_ptr, remaining, field.count)) {
                return false;
            }
        }

        uint8_t endian_flag = 0;
        if (!ros1_detail::read_primitive(data_ptr, remaining, endian_flag) ||
            !ros1_detail::read_primitive(data_ptr, remaining, point_step) ||
            !ros1_detail::read_primitive(data_ptr, remaining, row_step)) {
            return false;
        }
        is_bigendian = endian_flag != 0;

        uint32_t data_length = 0;
        if (!ros1_detail::read_primitive(data_ptr, remaining, data_length) || remaining < data_length) {
            return false;
        }
        data.assign(data_ptr, data_ptr + data_length);
        data_ptr += data_length;
        remaining -= data_length;

        uint8_t dense_flag = 0;
        if (!ros1_detail::read_primitive(data_ptr, remaining, dense_flag)) {
            return false;
        }
        is_dense = dense_flag != 0;

        return remaining == 0;
    }
};

struct ROS1CompressedImage {
    ROS1Header header;
    std::string format;
    std::vector<uint8_t> data;

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        uint32_t data_length = 0;
        if (!header.parse(data_ptr, remaining) ||
            !ros1_detail::read_string(data_ptr, remaining, format) ||
            !ros1_detail::read_primitive(data_ptr, remaining, data_length) ||
            remaining < data_length) {
            return false;
        }

        data.assign(data_ptr, data_ptr + data_length);
        data_ptr += data_length;
        remaining -= data_length;

        return remaining == 0;
    }
};

struct ROS1Vector3 {
    double x{};
    double y{};
    double z{};

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::read_primitive(data, remaining, x) &&
               ros1_detail::read_primitive(data, remaining, y) &&
               ros1_detail::read_primitive(data, remaining, z);
    }
};

struct ROS1Quaternion {
    double x{};
    double y{};
    double z{};
    double w{};

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::read_primitive(data, remaining, x) &&
               ros1_detail::read_primitive(data, remaining, y) &&
               ros1_detail::read_primitive(data, remaining, z) &&
               ros1_detail::read_primitive(data, remaining, w);
    }
};

struct ROS1Pose {
    ROS1Vector3 position;
    ROS1Quaternion orientation;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return position.parse(data, remaining) && orientation.parse(data, remaining);
    }
};

struct ROS1PoseWithCovariance {
    ROS1Pose pose;
    std::array<double, 36> covariance{};

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return pose.parse(data, remaining) &&
               ros1_detail::read_array(data, remaining, covariance);
    }
};

struct ROS1Twist {
    ROS1Vector3 linear;
    ROS1Vector3 angular;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return linear.parse(data, remaining) && angular.parse(data, remaining);
    }
};

struct ROS1TwistWithCovariance {
    ROS1Twist twist;
    std::array<double, 36> covariance{};

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return twist.parse(data, remaining) &&
               ros1_detail::read_array(data, remaining, covariance);
    }
};

struct ROS1PoseStamped {
    ROS1Header header;
    ROS1Pose pose;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return header.parse(data, remaining) &&
               pose.parse(data, remaining);
    }
};

struct ROS1Transform {
    ROS1Vector3 translation;
    ROS1Quaternion rotation;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return translation.parse(data, remaining) && rotation.parse(data, remaining);
    }
};

struct ROS1TransformStamped {
    ROS1Header header;
    std::string child_frame_id;
    ROS1Transform transform;

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return header.parse(data, remaining) &&
               ros1_detail::read_string(data, remaining, child_frame_id) &&
               transform.parse(data, remaining);
    }
};

struct ROS1TFMessage {
    std::vector<ROS1TransformStamped> transforms;

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        return ros1_detail::read_sequence(data_ptr, remaining, transforms) && remaining == 0;
    }
};

struct ROS1Path {
    ROS1Header header;
    std::vector<ROS1PoseStamped> poses;

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        if (!header.parse(data_ptr, remaining) ||
            !ros1_detail::read_sequence(data_ptr, remaining, poses)) {
            return false;
        }
        return remaining == 0;
    }
};

struct ROS1LivoxCustomPoint {
    uint32_t offset_time{};
    float x{};
    float y{};
    float z{};
    uint8_t reflectivity{};
    uint8_t tag{};
    uint8_t line{};

    bool parse(const uint8_t*& data, size_t& remaining)
    {
        return ros1_detail::read_primitive(data, remaining, offset_time) &&
               ros1_detail::read_primitive(data, remaining, x) &&
               ros1_detail::read_primitive(data, remaining, y) &&
               ros1_detail::read_primitive(data, remaining, z) &&
               ros1_detail::read_primitive(data, remaining, reflectivity) &&
               ros1_detail::read_primitive(data, remaining, tag) &&
               ros1_detail::read_primitive(data, remaining, line);
    }
};

struct ROS1LivoxCustomMsg {
    ROS1Header header;
    uint64_t timebase{};
    uint32_t point_num{};
    uint8_t lidar_id{};
    std::array<uint8_t, 3> reserved{};
    std::vector<ROS1LivoxCustomPoint> points;

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        uint32_t serialized_point_count = 0;
        if (!header.parse(data_ptr, remaining) ||
            !ros1_detail::read_primitive(data_ptr, remaining, timebase) ||
            !ros1_detail::read_primitive(data_ptr, remaining, point_num) ||
            !ros1_detail::read_primitive(data_ptr, remaining, lidar_id) ||
            !ros1_detail::read_array(data_ptr, remaining, reserved) ||
            !ros1_detail::read_primitive(data_ptr, remaining, serialized_point_count)) {
            return false;
        }

        points.resize(serialized_point_count);
        for (auto& point : points) {
            if (!point.parse(data_ptr, remaining)) {
                return false;
            }
        }

        if (point_num != serialized_point_count) {
            point_num = serialized_point_count;
        }
        return remaining == 0;
    }
};

struct ROS1Imu {
    ROS1Header header;
    ROS1Quaternion orientation;
    std::array<double, 9> orientation_covariance{};
    ROS1Vector3 angular_velocity;
    std::array<double, 9> angular_velocity_covariance{};
    ROS1Vector3 linear_acceleration;
    std::array<double, 9> linear_acceleration_covariance{};

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        return header.parse(data_ptr, remaining) &&
               orientation.parse(data_ptr, remaining) &&
               ros1_detail::read_array(data_ptr, remaining, orientation_covariance) &&
               angular_velocity.parse(data_ptr, remaining) &&
               ros1_detail::read_array(data_ptr, remaining, angular_velocity_covariance) &&
               linear_acceleration.parse(data_ptr, remaining) &&
               ros1_detail::read_array(data_ptr, remaining, linear_acceleration_covariance) &&
               remaining == 0;
    }
};

struct ROS1Odometry {
    ROS1Header header;
    std::string child_frame_id;
    ROS1Pose pose;
    std::array<double, 36> pose_covariance{};
    ROS1Twist twist;
    std::array<double, 36> twist_covariance{};

    bool parse(const uint8_t* msg_data, size_t msg_size)
    {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        ROS1PoseWithCovariance pose_with_cov;
        ROS1TwistWithCovariance twist_with_cov;

        if (!header.parse(data_ptr, remaining) ||
            !ros1_detail::read_string(data_ptr, remaining, child_frame_id) ||
            !pose_with_cov.parse(data_ptr, remaining) ||
            !twist_with_cov.parse(data_ptr, remaining)) {
            return false;
        }

        pose = pose_with_cov.pose;
        pose_covariance = pose_with_cov.covariance;
        twist = twist_with_cov.twist;
        twist_covariance = twist_with_cov.covariance;

        return remaining == 0;
    }
};
