#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// ROS1消息解析辅助结构
// ============================================================================

// ROS1 Header结构
struct ROS1Header {
    uint32_t seq;
    uint32_t stamp_sec;
    uint32_t stamp_nsec;
    std::string frame_id;

    bool parse(const uint8_t*& data, size_t& remaining) {
        if (remaining < 12) return false;

        seq = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        stamp_sec = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        stamp_nsec = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        // 读取frame_id字符串长度
        if (remaining < 4) return false;
        uint32_t str_len = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        if (remaining < str_len) return false;
        frame_id = std::string(reinterpret_cast<const char*>(data), str_len);
        data += str_len; remaining -= str_len;

        return true;
    }
};

// ROS1 PointCloud2字段
struct ROS1PointField {
    std::string name;
    uint32_t offset;
    uint8_t datatype;
    uint32_t count;
};

// ROS1 PointCloud2消息
struct ROS1PointCloud2 {
    ROS1Header header;
    uint32_t height;
    uint32_t width;
    std::vector<ROS1PointField> fields;
    bool is_bigendian;
    uint32_t point_step;
    uint32_t row_step;
    std::vector<uint8_t> data;
    bool is_dense;

    bool parse(const uint8_t* msg_data, size_t msg_size) {
        const uint8_t* data = msg_data;
        size_t remaining = msg_size;

        // 解析header
        if (!header.parse(data, remaining)) return false;

        // 解析height, width
        if (remaining < 8) return false;
        height = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        width = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        // 解析fields数组
        if (remaining < 4) return false;
        uint32_t fields_len = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        fields.resize(fields_len);
        for (uint32_t i = 0; i < fields_len; ++i) {
            // name
            if (remaining < 4) return false;
            uint32_t name_len = *reinterpret_cast<const uint32_t*>(data);
            data += 4; remaining -= 4;

            if (remaining < name_len) return false;
            fields[i].name = std::string(reinterpret_cast<const char*>(data), name_len);
            data += name_len; remaining -= name_len;

            // offset, datatype, count
            if (remaining < 9) return false;
            fields[i].offset = *reinterpret_cast<const uint32_t*>(data);
            data += 4; remaining -= 4;

            fields[i].datatype = *data;
            data += 1; remaining -= 1;

            fields[i].count = *reinterpret_cast<const uint32_t*>(data);
            data += 4; remaining -= 4;
        }

        // is_bigendian, point_step, row_step
        if (remaining < 9) return false;
        is_bigendian = *data != 0;
        data += 1; remaining -= 1;

        point_step = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        row_step = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        // data数组
        if (remaining < 4) return false;
        uint32_t data_len = *reinterpret_cast<const uint32_t*>(data);
        data += 4; remaining -= 4;

        if (remaining < data_len) return false;
        this->data.assign(data, data + data_len);
        data += data_len; remaining -= data_len;

        // is_dense
        if (remaining < 1) return false;
        is_dense = *data != 0;

        return true;
    }
};

// ROS1 CompressedImage消息
struct ROS1CompressedImage {
    ROS1Header header;
    std::string format;
    std::vector<uint8_t> data;

    bool parse(const uint8_t* msg_data, size_t msg_size) {
        const uint8_t* data_ptr = msg_data;
        size_t remaining = msg_size;

        // 解析header
        if (!header.parse(data_ptr, remaining)) return false;

        // 解析format字符串
        if (remaining < 4) return false;
        uint32_t format_len = *reinterpret_cast<const uint32_t*>(data_ptr);
        data_ptr += 4; remaining -= 4;

        if (remaining < format_len) return false;
        format = std::string(reinterpret_cast<const char*>(data_ptr), format_len);
        data_ptr += format_len; remaining -= format_len;

        // 解析data数组
        if (remaining < 4) return false;
        uint32_t data_len = *reinterpret_cast<const uint32_t*>(data_ptr);
        data_ptr += 4; remaining -= 4;

        if (remaining < data_len) return false;
        data.assign(data_ptr, data_ptr + data_len);

        return true;
    }
};

// ROS1 IMU消息（简化版）
struct ROS1Imu {
    ROS1Header header;
    struct Quaternion {
        double x, y, z, w;
    } orientation;
    double orientation_covariance[9];
    struct Vector3 {
        double x, y, z;
    } angular_velocity;
    double angular_velocity_covariance[9];
    struct Vector3 linear_acceleration;
    double linear_acceleration_covariance[9];

    bool parse(const uint8_t* msg_data, size_t msg_size) {
        const uint8_t* data = msg_data;
        size_t remaining = msg_size;

        // 解析header
        if (!header.parse(data, remaining)) return false;

        // 解析orientation quaternion
        if (remaining < 32) return false;  // 4 * 8 bytes
        orientation.x = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        orientation.y = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        orientation.z = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        orientation.w = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;

        // 解析orientation_covariance
        if (remaining < 72) return false;  // 9 * 8 bytes
        for (int i = 0; i < 9; ++i) {
            orientation_covariance[i] = *reinterpret_cast<const double*>(data);
            data += 8; remaining -= 8;
        }

        // 解析angular_velocity
        if (remaining < 24) return false;  // 3 * 8 bytes
        angular_velocity.x = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        angular_velocity.y = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        angular_velocity.z = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;

        // 解析angular_velocity_covariance
        if (remaining < 72) return false;
        for (int i = 0; i < 9; ++i) {
            angular_velocity_covariance[i] = *reinterpret_cast<const double*>(data);
            data += 8; remaining -= 8;
        }

        // 解析linear_acceleration
        if (remaining < 24) return false;
        linear_acceleration.x = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        linear_acceleration.y = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;
        linear_acceleration.z = *reinterpret_cast<const double*>(data);
        data += 8; remaining -= 8;

        // 解析linear_acceleration_covariance
        if (remaining < 72) return false;
        for (int i = 0; i < 9; ++i) {
            linear_acceleration_covariance[i] = *reinterpret_cast<const double*>(data);
            data += 8; remaining -= 8;
        }

        return true;
    }
};