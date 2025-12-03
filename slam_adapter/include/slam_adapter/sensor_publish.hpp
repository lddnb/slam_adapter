#pragma once

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <span>
#include <type_traits>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <spdlog/spdlog.h>

#include <slam_common/sensor_struct.hpp>
#include <slam_core/filter_state.hpp>
#include <slam_core/image.hpp>
#include <slam_core/point_cloud.hpp>
#include <slam_core/point_types.hpp>

using State = ms_slam::slam_core::FilterState;
using States = ms_slam::slam_core::FilterStates;

namespace ms_slam::slam_adapter
{
/**
 * @brief 坐标变换描述，供 TF 发布使用
 */
struct FrameTransformData {
    double timestamp{0.0};                                        ///< 时间戳（秒）
    std::string_view parent_frame;                                ///< 父坐标系
    std::string_view child_frame;                                 ///< 子坐标系
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};         ///< 平移量
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};  ///< 旋转量
};

namespace detail
{
/**
 * @brief 将秒转换为纳秒并写入头部
 * @param timestamp_sec 输入的秒
 * @param header 目标头部
 */
inline void FillTimestampHeader(double timestamp_sec, slam_common::TimeFrameHeader& header)
{
    header.timestamp_ns = static_cast<uint64_t>(std::llround(timestamp_sec * 1e9));
}

/**
 * @brief 拷贝字符串到定长数组，自动补 '\0'
 * @tparam N 目标数组长度
 * @param value 输入字符串
 * @param buffer 目标数组
 */
template <std::size_t N>
inline void CopyString(std::string_view value, std::array<char, N>& buffer)
{
    buffer.fill('\0');
    const std::size_t copy_len = std::min<std::size_t>(value.size(), N - 1);
    std::memcpy(buffer.data(), value.data(), copy_len);
}

/**
 * @brief 填充 3D 位姿到 Pose3d
 * @param translation 平移
 * @param rotation 旋转
 * @param target 目标 Pose3d
 */
inline void FillPose(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation, slam_common::Pose3d& target)
{
    target.position = {translation.x(), translation.y(), translation.z()};
    target.orientation = {rotation.x(), rotation.y(), rotation.z(), rotation.w()};
}
}  // namespace detail

/**
 * @brief 将单帧状态转换为 OdomData
 * @param state 输入滤波状态
 * @param frame_id 世界坐标系
 * @param child_frame_id 机体坐标系
 * @param message 输出里程计
 * @return 成功返回 true
 */
inline bool BuildOdomData(const State& state,
                          std::string_view frame_id,
                          std::string_view child_frame_id,
                          slam_common::OdomData& message)
{
    detail::FillTimestampHeader(state.timestamp(), message.header);
    detail::CopyString(frame_id, message.header.frame_id);
    detail::CopyString(child_frame_id, message.child_frame_id);

    detail::FillPose(state.p(), state.quat().normalized(), message.pose);

    const auto cov = state.cov();
    const int cov_rows = static_cast<int>(cov.rows());
    const int cov_cols = static_cast<int>(cov.cols());
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            const int idx = r * 6 + c;
            if (r < cov_rows && c < cov_cols) {
                message.pose_covariance[static_cast<std::size_t>(idx)] = cov(r, c);
            } else {
                message.pose_covariance[static_cast<std::size_t>(idx)] = 0.0;
            }
        }
    }

    const auto velocity = state.v();
    message.linear_velocity = {velocity.x(), velocity.y(), velocity.z()};
    message.angular_velocity = {0.0, 0.0, 0.0};  // 状态中未直接提供角速度，暂置零
    message.twist_covariance.fill(0.0);
    return true;
}

/**
 * @brief 将状态序列转换为 PathData
 * @param states 状态序列
 * @param frame_id 坐标系 ID
 * @param message 输出路径
 * @return 成功返回 true
 */
inline bool BuildPathData(const std::vector<State>& states, std::string_view frame_id, slam_common::PathData& message)
{
    if (states.empty()) {
        spdlog::warn("BuildPathData: empty states");
        return false;
    }

    detail::FillTimestampHeader(states.back().timestamp(), message.header);
    detail::CopyString(frame_id, message.header.frame_id);

    const std::size_t count = std::min<std::size_t>(states.size(), slam_common::kMaxPathPoses);
    message.pose_count = static_cast<uint32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& state = states[i];
        auto& pose = message.poses[i];
        pose.timestamp_ns = static_cast<uint64_t>(std::llround(state.timestamp() * 1e9));
        detail::FillPose(state.p(), state.quat().normalized(), pose.pose);
    }
    return true;
}

/**
 * @brief 构造坐标变换数组
 * @param transforms 变换列表
 * @param message 输出 TF 数组
 * @return 成功返回 true
 */
inline bool BuildFrameTransformArray(std::span<const FrameTransformData> transforms, slam_common::FrameTransformArray& message)
{
    if (transforms.empty()) {
        spdlog::warn("BuildFrameTransformArray: empty input");
        return false;
    }

    const std::size_t count = std::min<std::size_t>(transforms.size(), slam_common::kMaxFrameTransforms);
    message.transform_count = static_cast<uint32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& src = transforms[i];
        auto& dst = message.transforms[i];
        dst.timestamp_ns = static_cast<uint64_t>(std::llround(src.timestamp * 1e9));
        detail::CopyString(src.parent_frame, dst.parent_frame_id);
        detail::CopyString(src.child_frame, dst.child_frame_id);
        detail::FillPose(src.translation, src.rotation.normalized(), dst.transform);
    }
    return true;
}

/**
 * @brief 将 slam_core 图像封装为定长图像结构
 * @tparam ImageStruct 目标图像类型
 * @param image 输入图像
 * @param frame_id 坐标系 ID
 * @param encoding 编码字符串（如 bgr8）
 * @param message 输出结构
 * @return 成功返回 true
 */
template <typename ImageStruct>
inline bool BuildImageMessage(const slam_core::Image& image,
                              std::string_view frame_id,
                              std::string_view encoding,
                              ImageStruct& message)
{
    static_assert(std::is_same_v<ImageStruct, slam_common::Image>, "Unsupported image struct");

    const auto& mat = image.data();
    if (mat.empty() || mat.channels() != 3 || mat.type() != CV_8UC3) {
        spdlog::warn("BuildImageMessage: unsupported image format type={}, channels={}", mat.type(), mat.channels());
        return false;
    }

    const int width = mat.cols;
    const int height = mat.rows;
    constexpr int kChannels = 3;
    const std::size_t payload_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * kChannels;
    if (payload_size > message.data.size()) {
        spdlog::warn("BuildImageMessage: payload overflow, need {}, buffer {}", payload_size, message.data.size());
        return false;
    }

    message.header.timestamp_ns = static_cast<uint64_t>(std::llround(image.timestamp() * 1e9));
    detail::CopyString(frame_id, message.header.frame_id);
    detail::CopyString(encoding, message.header.encoding);
    message.header.width = static_cast<uint32_t>(width);
    message.header.height = static_cast<uint32_t>(height);
    message.header.step = static_cast<uint32_t>(width * kChannels);
    message.header.payload_size = static_cast<uint32_t>(payload_size);
    message.header.compressed = false;

    std::memcpy(message.data.data(), mat.data, payload_size);
    return true;
}

/**
 * @brief 将通用点云转换为 Mid360Frame（用于 map/local map 发布）
 * @tparam PointCloudT 点云类型
 * @param cloud 输入点云
 * @param frame_timestamp_ns 帧时间戳，若点云自带时间戳可为 0
 * @param frame_id 坐标系 ID
 * @param frame 输出 Mid360 帧
 * @return 成功返回 true
 */
template <typename PointCloudT>
inline bool BuildMid360FrameFromPointCloud(const PointCloudT& cloud,
                                           uint64_t frame_timestamp_ns,
                                           slam_common::Mid360Frame& frame,
                                           std::string_view frame_id = "")
{
    using Descriptor = typename PointCloudT::descriptor_type;
    static_assert(slam_core::has_field_v<slam_core::PositionTag, Descriptor>, "Point cloud descriptor must provide PositionTag");

    constexpr bool kHasIntensity = slam_core::has_field_v<slam_core::IntensityTag, Descriptor>;
    constexpr bool kHasTimestamp = slam_core::has_field_v<slam_core::TimestampTag, Descriptor>;

    const std::size_t point_count = std::min<std::size_t>(cloud.size(), slam_common::kMid360MaxPoints);
    if (point_count == 0) {
        spdlog::warn("BuildMid360FrameFromPointCloud: empty cloud");
        frame.point_count = 0;
        detail::CopyString(frame_id, frame.frame_id);
        return false;
    }

    uint64_t base_ts_ns = frame_timestamp_ns;
    if constexpr (kHasTimestamp) {
        if (base_ts_ns == 0 && !cloud.empty()) {
            const double ts_sec = cloud.timestamp(0);
            if (std::isfinite(ts_sec) && ts_sec > 0.0) {
                base_ts_ns = static_cast<uint64_t>(std::llround(ts_sec * 1e9));
            }
        }
    }

    frame.index = 0;
    frame.frame_timestamp_ns = base_ts_ns;
    frame.point_count = static_cast<uint32_t>(point_count);
    detail::CopyString(frame_id, frame.frame_id);

    for (std::size_t i = 0; i < point_count; ++i) {
        const auto pos = cloud.position(i);
        auto& dst = frame.points[i];
        dst.x = static_cast<float>(pos.x());
        dst.y = static_cast<float>(pos.y());
        dst.z = static_cast<float>(pos.z());

        if constexpr (kHasIntensity) {
            const float intensity = static_cast<float>(cloud.intensity(i));
            dst.intensity = static_cast<uint8_t>(std::clamp(intensity, 0.0F, 255.0F));
        } else {
            dst.intensity = 0;
        }

        dst.tag = 0;

        if constexpr (kHasTimestamp) {
            const double ts_sec = cloud.timestamp(i);
            const double ts_ns = ts_sec * 1e9;
            if (std::isfinite(ts_ns) && ts_ns > 0.0) {
                dst.timestamp_ns = static_cast<uint64_t>(std::llround(ts_ns));
            } else {
                dst.timestamp_ns = base_ts_ns;
            }
        } else {
            dst.timestamp_ns = base_ts_ns;
        }
    }
    return true;
}

}  // namespace ms_slam::slam_adapter
