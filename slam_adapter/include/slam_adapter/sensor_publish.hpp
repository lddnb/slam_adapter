#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <foxglove/CompressedImage.pb.h>
#include <foxglove/FrameTransforms.pb.h>
#include <foxglove/PointCloud.pb.h>
#include <foxglove/PoseInFrame.pb.h>
#include <foxglove/PosesInFrame.pb.h>
#include <foxglove/SceneUpdate.pb.h>

#include <slam_common/foxglove_messages.hpp>
#include <slam_core/image.hpp>
#include <slam_core/point_cloud.hpp>
#include <slam_core/point_types.hpp>
#include <slam_core/filter_state.hpp>

using State = ms_slam::slam_core::FilterState;
using States = ms_slam::slam_core::FilterStates;

namespace ms_slam::slam_adapter
{
/**
 * @brief Foxglove 坐标变换描述
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
 * @brief 将秒转换为 Protobuf Timestamp
 * @param timestamp 输入的秒
 * @param target 目标时间戳
 */
inline void FillTimestamp(const double timestamp, google::protobuf::Timestamp& target)
{
    const auto seconds = static_cast<std::int64_t>(timestamp);
    const auto nanos = static_cast<std::int32_t>(std::round((timestamp - static_cast<double>(seconds)) * 1e9));
    target.set_seconds(seconds);
    target.set_nanos(nanos);
}

/**
 * @brief 填充向量字段
 * @param vec Eigen 向量
 * @param target Protobuf Vector3
 */
inline void FillVector3(const Eigen::Vector3d& vec, foxglove::Vector3& target)
{
    target.set_x(vec.x());
    target.set_y(vec.y());
    target.set_z(vec.z());
}

/**
 * @brief 填充四元数字段
 * @param quat Eigen 四元数
 * @param target Protobuf Quaternion
 */
inline void FillQuaternion(const Eigen::Quaterniond& quat, foxglove::Quaternion& target)
{
    target.set_x(quat.x());
    target.set_y(quat.y());
    target.set_z(quat.z());
    target.set_w(quat.w());
}

/**
 * @brief 计算位置协方差对应的椭球参数
 * @param position_cov 位置协方差
 * @param axes 输出椭球长轴
 * @param orientation 输出椭球朝向
 * @return 成功返回 true
 */
inline bool ComputePositionCovarianceEllipsoid(const Eigen::Matrix3d& position_cov, Eigen::Vector3d& axes, Eigen::Quaterniond& orientation)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(position_cov);
    if (eigen_solver.info() != Eigen::Success) {
        spdlog::warn("ComputePositionCovarianceEllipsoid: eigen decomposition failed");
        return false;
    }

    constexpr double kMinEigenValue = 1e-12;
    constexpr double kScaleSigma = 3.0;

    const Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues().cwiseMax(kMinEigenValue);
    axes = eigenvalues.cwiseSqrt() * kScaleSigma;

    Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();
    if (eigenvectors.determinant() < 0.0) {
        eigenvectors.col(0) = -eigenvectors.col(0);
    }

    orientation = Eigen::Quaterniond(eigenvectors);
    orientation.normalize();
    return true;
}

/**
 * @brief 写入 float 数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreFloat(std::uint8_t* destination, const float value)
{
    std::memcpy(destination, std::addressof(value), sizeof(float));
}

/**
 * @brief 写入 double 数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreDouble(std::uint8_t* destination, const double value)
{
    std::memcpy(destination, std::addressof(value), sizeof(double));
}

/**
 * @brief 写入 uint8_t 数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreU8(std::uint8_t* destination, const std::uint8_t value)
{
    *destination = value;
}

/**
 * @brief 填充单位姿态
 * @param pose 目标 Pose
 */
inline void FillIdentityPose(foxglove::Pose& pose)
{
    auto* position = pose.mutable_position();
    position->set_x(0.0);
    position->set_y(0.0);
    position->set_z(0.0);

    auto* orientation = pose.mutable_orientation();
    orientation->set_x(0.0);
    orientation->set_y(0.0);
    orientation->set_z(0.0);
    orientation->set_w(1.0);
}

}  // namespace detail

/**
 * @brief 将 State 转换为 Foxglove PoseInFrame 消息
 * @param state 里程计状态
 * @param frame_id 坐标系名称
 * @param message 目标消息
 * @return 成功返回 true
 */
inline bool BuildFoxglovePoseInFrame(const State& state, std::string_view frame_id, foxglove::PoseInFrame& message)
{
    message.Clear();

    detail::FillTimestamp(state.timestamp(), *message.mutable_timestamp());
    message.set_frame_id(frame_id.data(), static_cast<int>(frame_id.size()));

    auto* pose = message.mutable_pose();
    detail::FillVector3(state.p(), *pose->mutable_position());
    detail::FillQuaternion(state.quat(), *pose->mutable_orientation());
    return true;
}

/**
 * @brief 将状态序列转换为 Foxglove PosesInFrame 消息
 * @param states 状态序列
 * @param frame_id 坐标系名称
 * @param message 目标消息
 * @return 成功返回 true
 */
inline bool BuildFoxglovePosesInFrame(const std::vector<State>& states, std::string_view frame_id, foxglove::PosesInFrame& message)
{
    message.Clear();

    if (states.empty()) {
        return false;
    }

    detail::FillTimestamp(states.back().timestamp(), *message.mutable_timestamp());
    message.set_frame_id(frame_id.data(), static_cast<int>(frame_id.size()));

    for (const auto& state : states) {
        auto* pose = message.add_poses();
        detail::FillVector3(state.p(), *pose->mutable_position());
        detail::FillQuaternion(state.quat().normalized(), *pose->mutable_orientation());
    }

    return true;
}

/**
 * @brief 构建 Foxglove FrameTransforms 消息
 * @param transforms 变换列表
 * @param message 目标消息
 * @return 成功返回 true
 */
inline bool BuildFoxgloveFrameTransforms(std::span<const FrameTransformData> transforms, foxglove::FrameTransforms& message)
{
    message.Clear();

    if (transforms.empty()) {
        spdlog::warn("BuildFoxgloveFrameTransforms: empty transform list");
        return false;
    }

    for (const auto& transform : transforms) {
        auto* tf = message.add_transforms();
        detail::FillTimestamp(transform.timestamp, *tf->mutable_timestamp());
        tf->set_parent_frame_id(transform.parent_frame.data(), static_cast<int>(transform.parent_frame.size()));
        tf->set_child_frame_id(transform.child_frame.data(), static_cast<int>(transform.child_frame.size()));
        detail::FillVector3(transform.translation, *tf->mutable_translation());
        detail::FillQuaternion(transform.rotation.normalized(), *tf->mutable_rotation());
    }

    return true;
}

/**
 * @brief 基于 State 构建 Foxglove SceneUpdate 椭球消息
 * @param state 当前状态
 * @param frame_id 所属坐标系
 * @param entity_id 场景实体 ID
 * @param message 目标消息
 * @return 成功返回 true
 */
inline bool BuildFoxgloveSceneUpdateFromState(
    const State& state,
    std::string_view frame_id,
    std::string_view entity_id,
    foxglove::SceneUpdate& message)
{
    message.Clear();

    Eigen::Vector3d axes{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    const Eigen::Matrix3d position_cov = state.cov().block<3, 3>(0, 0);
    if (!detail::ComputePositionCovarianceEllipsoid(position_cov, axes, orientation)) {
        return false;
    }

    auto* entity = message.add_entities();
    detail::FillTimestamp(state.timestamp(), *entity->mutable_timestamp());
    entity->set_frame_id(frame_id.data(), static_cast<int>(frame_id.size()));
    entity->set_id(entity_id.data(), static_cast<int>(entity_id.size()));

    auto* sphere = entity->add_spheres();
    auto* pose = sphere->mutable_pose();
    detail::FillVector3(state.p(), *pose->mutable_position());
    detail::FillQuaternion(orientation, *pose->mutable_orientation());
    detail::FillVector3(axes * 2.0, *sphere->mutable_size());

    auto* color = sphere->mutable_color();
    color->set_r(0.2);
    color->set_g(0.6);
    color->set_b(0.9);
    color->set_a(0.35);

    return true;
}

/**
 * @brief 将点云转换为 Foxglove PointCloud 消息
 * @tparam PointCloudT 点云类型
 * @param cloud 点云数据
 * @param frame_id 坐标系名称
 * @param message 目标消息
 * @return 成功返回 true
 */
template <typename PointCloudT>
inline bool BuildFoxglovePointCloud(const PointCloudT& cloud, std::string_view frame_id, foxglove::PointCloud& message)
{
    message.Clear();

    using Descriptor = typename PointCloudT::descriptor_type;
    static_assert(slam_core::has_field_v<slam_core::PositionTag, Descriptor>, "Point cloud descriptor must provide PositionTag");

    constexpr bool kHasIntensity = slam_core::has_field_v<slam_core::IntensityTag, Descriptor>;
    constexpr bool kHasRGB = slam_core::has_field_v<slam_core::RGBTag, Descriptor>;
    constexpr bool kHasTimestamp = slam_core::has_field_v<slam_core::TimestampTag, Descriptor>;

    struct FieldInfo {
        const char* name;
        std::uint32_t offset;
        std::uint32_t size;
        foxglove::PackedElementField::NumericType numeric_type;
    };

    constexpr std::size_t kFieldCount =
        3 + static_cast<std::size_t>(kHasIntensity) + static_cast<std::size_t>(kHasRGB) * 3 + static_cast<std::size_t>(kHasTimestamp);

    std::array<FieldInfo, kFieldCount> field_infos{};
    std::uint32_t current_offset = 0;
    std::size_t field_index = 0;

    auto push_field = [&](const char* name, std::uint32_t size, foxglove::PackedElementField::NumericType type) {
        field_infos[field_index++] = FieldInfo{.name = name, .offset = current_offset, .size = size, .numeric_type = type};
        current_offset += size;
    };

    push_field("x", sizeof(float), foxglove::PackedElementField::FLOAT32);
    push_field("y", sizeof(float), foxglove::PackedElementField::FLOAT32);
    push_field("z", sizeof(float), foxglove::PackedElementField::FLOAT32);

    if constexpr (kHasIntensity) {
        push_field("intensity", sizeof(float), foxglove::PackedElementField::FLOAT32);
    }

    if constexpr (kHasRGB) {
        push_field("red", sizeof(std::uint8_t), foxglove::PackedElementField::UINT8);
        push_field("green", sizeof(std::uint8_t), foxglove::PackedElementField::UINT8);
        push_field("blue", sizeof(std::uint8_t), foxglove::PackedElementField::UINT8);
    }

    if constexpr (kHasTimestamp) {
        push_field("timestamp", sizeof(double), foxglove::PackedElementField::FLOAT64);
    }

    const std::uint32_t point_stride = current_offset;
    const std::size_t point_count = cloud.size();
    std::vector<std::uint8_t> data(point_count * point_stride);

    for (std::size_t idx = 0; idx < point_count; ++idx) {
        std::uint8_t* base = data.data() + idx * point_stride;
        const auto position = cloud.position(idx);

        detail::StoreFloat(base + field_infos[0].offset, static_cast<float>(position.x()));
        detail::StoreFloat(base + field_infos[1].offset, static_cast<float>(position.y()));
        detail::StoreFloat(base + field_infos[2].offset, static_cast<float>(position.z()));

        std::size_t writer_index = 3;

        if constexpr (kHasIntensity) {
            detail::StoreFloat(base + field_infos[writer_index].offset, static_cast<float>(cloud.intensity(idx)));
            ++writer_index;
        }

        if constexpr (kHasRGB) {
            const auto rgb = cloud.rgb(idx);
            detail::StoreU8(base + field_infos[writer_index + 0].offset, static_cast<std::uint8_t>(rgb(0)));
            detail::StoreU8(base + field_infos[writer_index + 1].offset, static_cast<std::uint8_t>(rgb(1)));
            detail::StoreU8(base + field_infos[writer_index + 2].offset, static_cast<std::uint8_t>(rgb(2)));
            writer_index += 3;
        }

        if constexpr (kHasTimestamp) {
            detail::StoreDouble(base + field_infos[writer_index].offset, static_cast<double>(cloud.timestamp(idx)));
        }
    }

    double message_time = 0.0;
    if constexpr (kHasTimestamp) {
        message_time = point_count > 0 ? static_cast<double>(cloud.timestamp(0)) : 0.0;
    }

    detail::FillTimestamp(message_time, *message.mutable_timestamp());
    message.set_frame_id(frame_id.data(), static_cast<int>(frame_id.size()));
    message.set_point_stride(point_stride);
    message.set_data(reinterpret_cast<const char*>(data.data()), static_cast<int>(data.size()));

    auto* pose = message.mutable_pose();
    detail::FillIdentityPose(*pose);

    for (const auto& field : field_infos) {
        auto* proto_field = message.add_fields();
        proto_field->set_name(field.name);
        proto_field->set_offset(field.offset);
        proto_field->set_type(field.numeric_type);
    }

    return true;
}

/**
 * @brief 将图像转换为 Foxglove CompressedImage 消息
 * @param image 图像数据
 * @param frame_id 坐标系名称
 * @param format 压缩格式（jpeg/png/webp）
 * @param quality 压缩质量
 * @param message 目标消息
 * @return 成功返回 true
 */
inline bool BuildFoxgloveCompressedImage(
    const slam_core::Image& image,
    std::string_view frame_id,
    std::string_view format,
    int quality,
    foxglove::CompressedImage& message)
{
    message.Clear();

    if (image.data().empty()) {
        spdlog::warn("BuildFoxgloveCompressedImage: empty image data");
        return false;
    }

    const int clamped_quality = std::clamp(quality, 1, 100);
    std::vector<int> params;
    std::string extension = "." + std::string(format);

    if (format == "jpeg" || format == "jpg") {
        params = {cv::IMWRITE_JPEG_QUALITY, clamped_quality};
        extension = ".jpg";
    } else if (format == "png") {
        params = {cv::IMWRITE_PNG_COMPRESSION, std::clamp(9 - clamped_quality / 12, 0, 9)};
        extension = ".png";
    } else if (format == "webp") {
        params = {cv::IMWRITE_WEBP_QUALITY, clamped_quality};
        extension = ".webp";
    }

    std::vector<std::uint8_t> compressed;
    if (!cv::imencode(extension, image.data(), compressed, params)) {
        spdlog::error("BuildFoxgloveCompressedImage: failed to encode image with format {}", format);
        return false;
    }

    detail::FillTimestamp(image.timestamp(), *message.mutable_timestamp());
    message.set_frame_id(frame_id.data(), static_cast<int>(frame_id.size()));
    message.set_format(format.data(), static_cast<int>(format.size()));
    message.set_data(reinterpret_cast<const char*>(compressed.data()), static_cast<int>(compressed.size()));

    return true;
}

}  // namespace ms_slam::slam_adapter
