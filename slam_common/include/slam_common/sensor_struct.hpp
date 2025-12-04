#pragma once

#include <cstdint>

#include <array>
#include <string>
#include <type_traits>

#include <spdlog/spdlog.h>

namespace ms_slam::slam_common
{
/// @brief 坐标系 ID 最大长度（包含结尾的 '\0'）
inline constexpr std::size_t kFrameIdMaxLength = 64;

/// @brief 图像编码字符串最大长度（包含结尾的 '\0'）
inline constexpr std::size_t kImageEncodingMaxLength = 32;

/**
 * @brief Livox IMU 定长数据结构，适用于零拷贝共享内存传输
 */
struct LivoxImuData {
    uint64_t timestamp_ns{0};                    ///< 统一纳秒时间戳（通常来源于硬件时钟）
    uint32_t index{0};                           ///< 帧序号或驱动提供的递增索引
    std::array<double, 3> angular_velocity{};     ///< 角速度 (rad/s)
    std::array<double, 3> linear_acceleration{};  ///< 线加速度 (m/s^2)
};

/// @brief Livox 单帧最大点数，固定 24000 便于零拷贝传输，Avia 点数为 24000，Mid360 点数为 20000
inline constexpr std::size_t kLivoxMaxPoints = 24000;

/**
 * @brief Livox 单点定长结构
 */
struct LivoxPoint {
    float x{0.0F};             ///< 点的 x 坐标（米）
    float y{0.0F};             ///< 点的 y 坐标（米）
    float z{0.0F};             ///< 点的 z 坐标（米）
    uint8_t intensity{0};      ///< 点的强度
    uint8_t tag{0};            ///< 设备标签或线号
    uint64_t timestamp_ns{0};  ///< 点级时间戳（纳秒）
};

/**
 * @brief Livox 点云定长帧
 */
struct LivoxPointCloudDate {
    uint32_t index{0};                                   ///< 帧序号
    uint32_t point_count{0};                             ///< 实际点数，<= kLivoxMaxPoints
    uint64_t frame_timestamp_ns{0};                      ///< 帧级时间戳（纳秒）
    std::array<char, kFrameIdMaxLength> frame_id{};      ///< 坐标系 ID（以 '\0' 结尾）
    std::array<LivoxPoint, kLivoxMaxPoints> points{};  ///< 点云定长数组
};

/**
 * @brief 时间戳 + 坐标系头部信息
 */
struct TimeFrameHeader {
    uint64_t timestamp_ns{0};                        ///< 统一纳秒时间戳
    std::array<char, kFrameIdMaxLength> frame_id{};  ///< 坐标系 ID（以 '\0' 结尾）
};

/**
 * @brief 三维姿态（位置 + 四元数）
 */
struct Pose3d {
    std::array<double, 3> position{0.0, 0.0, 0.0};          ///< 平移 (m)
    std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};  ///< 四元数 (x, y, z, w)
};

/**
 * @brief 定长图像头部
 */
struct ImageHeader {
    uint64_t timestamp_ns{0};                              ///< 时间戳（纳秒）
    std::array<char, kFrameIdMaxLength> frame_id{};        ///< 坐标系 ID
    std::array<char, kImageEncodingMaxLength> encoding{};  ///< 编码格式（如 rgb8/jpeg 等）
    uint32_t width{0};                                     ///< 图像宽度
    uint32_t height{0};                                    ///< 图像高度
    uint32_t step{0};                                      ///< 行跨度（未压缩时有效）
    uint32_t payload_size{0};                              ///< 实际数据长度
    bool compressed{false};                                ///< 是否为压缩数据
};

/// @brief 图像数据最大长度（3 通道）
inline constexpr std::size_t kImageMaxDataSize = 2592U * 1944U * 3U;

/**
 * @brief 图像定长结构
 */
struct ImageDate {
    ImageHeader header;                                  ///< 头部信息
    std::array<std::uint8_t, kImageMaxDataSize> data{};  ///< 图像数据
};

/// @brief 路径最大点数
inline constexpr std::size_t kMaxPathPoses = 2048;

/**
 * @brief 单个位姿节点
 */
struct PathPose {
    uint64_t timestamp_ns{0};  ///< 该节点时间戳
    Pose3d pose;               ///< 位置 + 朝向
};

/**
 * @brief 路径定长结构
 */
struct PathData {
    TimeFrameHeader header;                       ///< 路径头
    uint32_t pose_count{0};                       ///< 实际节点数量
    std::array<PathPose, kMaxPathPoses> poses{};  ///< 路径节点
};

/**
 * @brief 里程计数据（含位姿与速度）
 */
struct OdomData {
    TimeFrameHeader header;                                 ///< 里程计头
    std::array<char, kFrameIdMaxLength> child_frame_id{};   ///< 子坐标系
    Pose3d pose;                                            ///< 位姿
    std::array<double, 36> pose_covariance{};               ///< 位姿协方差
    std::array<double, 3> linear_velocity{0.0, 0.0, 0.0};   ///< 线速度 (m/s)
    std::array<double, 3> angular_velocity{0.0, 0.0, 0.0};  ///< 角速度 (rad/s)
    std::array<double, 36> twist_covariance{};              ///< 速度协方差
};

/// @brief TF 批处理最大数量
inline constexpr std::size_t kMaxFrameTransforms = 128;

/**
 * @brief 单个坐标系变换
 */
struct FrameTransform {
    uint64_t timestamp_ns{0};                               ///< 时间戳
    std::array<char, kFrameIdMaxLength> parent_frame_id{};  ///< 父坐标系
    std::array<char, kFrameIdMaxLength> child_frame_id{};   ///< 子坐标系
    Pose3d transform;                                       ///< 变换
};

/**
 * @brief TF 消息批处理
 */
struct FrameTransformArray {
    uint32_t transform_count{0};                                   ///< 变换数量
    std::array<FrameTransform, kMaxFrameTransforms> transforms{};  ///< 变换列表
};

static_assert(std::is_trivially_copyable_v<LivoxImuData>, "LivoxImuData must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<LivoxPoint>, "LivoxPoint must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<LivoxPointCloudDate>, "LivoxPointCloudDate must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<ImageHeader>, "ImageHeader must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<ImageDate>, "ImageDate must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<PathPose>, "PathPose must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<PathData>, "PathData must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<OdomData>, "OdomData must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<FrameTransform>, "FrameTransform must be trivially copyable for zero-copy transport");
static_assert(std::is_trivially_copyable_v<FrameTransformArray>, "FrameTransformArray must be trivially copyable for zero-copy transport");

}  // namespace ms_slam::slam_common
