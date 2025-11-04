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
#include <flatbuffers/flatbuffers.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <slam_common/foxglove_messages.hpp>
#include <slam_core/image.hpp>
#include <slam_core/point_cloud.hpp>
#include <slam_core/point_types.hpp>
#include <slam_core/state.hpp>

namespace ms_slam::slam_adapter
{

/**
 * @brief Foxglove坐标变换描述
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
 * @brief 将秒转换为Foxglove时间结构
 * @param timestamp 输入的秒
 * @return Foxglove时间结构
 */
inline foxglove::Time MakeFoxgloveTime(const double timestamp)
{
    const std::uint32_t sec32 = static_cast<std::uint32_t>(timestamp);
    const std::uint32_t nsec32 = static_cast<std::uint32_t>(std::round((timestamp - sec32) * 1e9));
    return foxglove::Time(sec32, nsec32);
}

/**
 * @brief 计算位置协方差对应的椭球参数
 * @param position_cov 位置协方差
 * @param axes 输出椭球长轴
 * @param orientation 输出椭球朝向
 * @return 成功返回true
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
 * @brief 生成身份姿态
 * @param builder FlatBuffer构造器
 * @return 身份姿态偏移量
 */
inline flatbuffers::Offset<foxglove::Pose> CreateIdentityPose(flatbuffers::FlatBufferBuilder& builder)
{
    const auto zero = foxglove::CreateVector3(builder, 0.0, 0.0, 0.0);
    const auto identity = foxglove::CreateQuaternion(builder, 0.0, 0.0, 0.0, 1.0);
    return foxglove::CreatePose(builder, zero, identity);
}

/**
 * @brief 快速写入float数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreFloat(std::uint8_t* destination, const float value)
{
    std::memcpy(destination, std::addressof(value), sizeof(float));
}

/**
 * @brief 快速写入double数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreDouble(std::uint8_t* destination, const double value)
{
    std::memcpy(destination, std::addressof(value), sizeof(double));
}

/**
 * @brief 快速写入uint8_t数据
 * @param destination 目标指针
 * @param value 写入值
 */
inline void StoreU8(std::uint8_t* destination, const std::uint8_t value)
{
    *destination = value;
}

}  // namespace detail

/**
 * @brief 将State转换为Foxglove PoseInFrame消息
 * @param state 里程计状态
 * @param frame_id 坐标系名称
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
inline bool BuildFoxglovePoseInFrame(const slam_core::State& state, std::string_view frame_id, flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    const Eigen::Vector3d position = state.p();
    const Eigen::Quaterniond orientation = state.quat();

    const foxglove::Time timestamp = detail::MakeFoxgloveTime(state.timestamp());
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto position_offset = foxglove::CreateVector3(builder, position.x(), position.y(), position.z());
    const auto orientation_offset = foxglove::CreateQuaternion(builder, orientation.x(), orientation.y(), orientation.z(), orientation.w());
    const auto pose_offset = foxglove::CreatePose(builder, position_offset, orientation_offset);

    const auto pose_in_frame_offset = foxglove::CreatePoseInFrame(builder, &timestamp, frame_id_offset, pose_offset);
    foxglove::FinishPoseInFrameBuffer(builder, pose_in_frame_offset);
    return true;
}

/**
 * @brief 将状态序列转换为Foxglove PosesInFrame消息
 * @param states 状态序列
 * @param frame_id 坐标系名称
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
inline bool BuildFoxglovePosesInFrame(const std::vector<slam_core::State>& states, std::string_view frame_id, flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    if (states.empty()) {
        // spdlog::warn("BuildFoxglovePosesInFrame: empty state sequence");
        return false;
    }

    std::vector<flatbuffers::Offset<foxglove::Pose>> pose_offsets;
    pose_offsets.reserve(states.size());

    for (const auto& state : states) {
        const Eigen::Vector3d position = state.p();
        const Eigen::Quaterniond orientation = state.quat().normalized();

        const auto position_offset = foxglove::CreateVector3(builder, position.x(), position.y(), position.z());
        const auto orientation_offset = foxglove::CreateQuaternion(builder, orientation.x(), orientation.y(), orientation.z(), orientation.w());
        pose_offsets.emplace_back(foxglove::CreatePose(builder, position_offset, orientation_offset));
    }

    const auto poses_vector = builder.CreateVector(pose_offsets);
    const foxglove::Time timestamp = detail::MakeFoxgloveTime(states.back().timestamp());
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());

    // 使用最新状态的时间戳作为路径消息的时间标记
    const auto poses_in_frame_offset = foxglove::CreatePosesInFrame(builder, &timestamp, frame_id_offset, poses_vector);
    foxglove::FinishPosesInFrameBuffer(builder, poses_in_frame_offset);
    return true;
}

/**
 * @brief 构建Foxglove FrameTransforms消息
 * @param transforms 变换列表
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
inline bool BuildFoxgloveFrameTransforms(std::span<const FrameTransformData> transforms, flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    if (transforms.empty()) {
        spdlog::warn("BuildFoxgloveFrameTransforms: empty transform list");
        return false;
    }

    std::vector<flatbuffers::Offset<foxglove::FrameTransform>> offsets;
    offsets.reserve(transforms.size());

    for (const auto& transform : transforms) {
        const Eigen::Quaterniond normalized = transform.rotation.normalized();
        const foxglove::Time timestamp = detail::MakeFoxgloveTime(transform.timestamp);

        const auto parent_offset = builder.CreateString(transform.parent_frame.data(), transform.parent_frame.size());
        const auto child_offset = builder.CreateString(transform.child_frame.data(), transform.child_frame.size());
        const auto translation_offset =
            foxglove::CreateVector3(builder, transform.translation.x(), transform.translation.y(), transform.translation.z());
        const auto rotation_offset = foxglove::CreateQuaternion(builder, normalized.x(), normalized.y(), normalized.z(), normalized.w());

        offsets.emplace_back(foxglove::CreateFrameTransform(builder, &timestamp, parent_offset, child_offset, translation_offset, rotation_offset));
    }

    const auto transforms_vector = builder.CreateVector(offsets);
    const auto frame_transforms_offset = foxglove::CreateFrameTransforms(builder, transforms_vector);
    foxglove::FinishFrameTransformsBuffer(builder, frame_transforms_offset);
    return true;
}

/**
 * @brief 基于State构建Foxglove SceneUpdate椭球消息
 * @param state 当前状态
 * @param frame_id 所属坐标系
 * @param entity_id 场景实体ID
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
inline bool BuildFoxgloveSceneUpdateFromState(
    const slam_core::State& state,
    std::string_view frame_id,
    std::string_view entity_id,
    flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    Eigen::Vector3d axes{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    const Eigen::Matrix3d position_cov = state.cov().block<3, 3>(0, 0);
    if (!detail::ComputePositionCovarianceEllipsoid(position_cov, axes, orientation)) {
        return false;
    }

    const foxglove::Time timestamp = detail::MakeFoxgloveTime(state.timestamp());
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto entity_id_offset = builder.CreateString(entity_id.data(), entity_id.size());

    const Eigen::Vector3d position = state.p();
    const auto position_offset = foxglove::CreateVector3(builder, position.x(), position.y(), position.z());
    const auto orientation_offset = foxglove::CreateQuaternion(builder, orientation.x(), orientation.y(), orientation.z(), orientation.w());
    const auto pose_offset = foxglove::CreatePose(builder, position_offset, orientation_offset);
    const auto axes_offset = foxglove::CreateVector3(builder, axes.x(), axes.y(), axes.z());
    const auto color_offset = foxglove::CreateColor(builder, 0.2, 0.6, 0.9, 0.35);

    const auto sphere_offset = foxglove::CreateSpherePrimitive(builder, pose_offset, axes_offset, color_offset);
    const auto spheres_vector = builder.CreateVector(&sphere_offset, 1);

    foxglove::SceneEntityBuilder entity_builder(builder);
    entity_builder.add_timestamp(&timestamp);
    entity_builder.add_frame_id(frame_id_offset);
    entity_builder.add_id(entity_id_offset);
    entity_builder.add_spheres(spheres_vector);
    const auto entity_offset = entity_builder.Finish();

    const auto entities_vector = builder.CreateVector(&entity_offset, 1);
    foxglove::SceneUpdateBuilder update_builder(builder);
    update_builder.add_entities(entities_vector);
    const auto update_offset = update_builder.Finish();

    foxglove::FinishSceneUpdateBuffer(builder, update_offset);
    return true;
}

/**
 * @brief 将任意包含位置场的点云转换为Foxglove PointCloud消息
 * @tparam PointCloudT 点云类型，需符合slam_core::PointCloud接口
 * @param cloud 点云数据
 * @param frame_id 坐标系名称
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
template <typename PointCloudT>
inline bool BuildFoxglovePointCloud(const PointCloudT& cloud, std::string_view frame_id, flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

    using Descriptor = typename PointCloudT::descriptor_type;
    static_assert(slam_core::has_field_v<slam_core::PositionTag, Descriptor>, "Point cloud descriptor must provide PositionTag");

    constexpr bool kHasIntensity = slam_core::has_field_v<slam_core::IntensityTag, Descriptor>;
    constexpr bool kHasRGB = slam_core::has_field_v<slam_core::RGBTag, Descriptor>;
    constexpr bool kHasTimestamp = slam_core::has_field_v<slam_core::TimestampTag, Descriptor>;

    struct FieldInfo {
        const char* name;
        std::uint32_t offset;
        std::uint32_t size;
        foxglove::NumericType numeric_type;
    };

    constexpr std::size_t kFieldCount =
        3 + static_cast<std::size_t>(kHasIntensity) + static_cast<std::size_t>(kHasRGB) * 3 + static_cast<std::size_t>(kHasTimestamp);

    std::array<FieldInfo, kFieldCount> field_infos{};
    std::uint32_t current_offset = 0;
    std::size_t field_index = 0;

    auto push_field = [&](const char* name, std::uint32_t size, foxglove::NumericType type) {
        field_infos[field_index++] = FieldInfo{.name = name, .offset = current_offset, .size = size, .numeric_type = type};
        current_offset += size;
    };

    push_field("x", sizeof(float), foxglove::NumericType_FLOAT32);
    push_field("y", sizeof(float), foxglove::NumericType_FLOAT32);
    push_field("z", sizeof(float), foxglove::NumericType_FLOAT32);

    if constexpr (kHasIntensity) {
        push_field("intensity", sizeof(float), foxglove::NumericType_FLOAT32);
    }

    if constexpr (kHasRGB) {
        push_field("red", sizeof(std::uint8_t), foxglove::NumericType_UINT8);
        push_field("green", sizeof(std::uint8_t), foxglove::NumericType_UINT8);
        push_field("blue", sizeof(std::uint8_t), foxglove::NumericType_UINT8);
    }

    if constexpr (kHasTimestamp) {
        push_field("timestamp", sizeof(double), foxglove::NumericType_FLOAT64);
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

    const foxglove::Time timestamp_struct = detail::MakeFoxgloveTime(message_time);
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());

    std::vector<flatbuffers::Offset<foxglove::PackedElementField>> field_offsets;
    field_offsets.reserve(field_infos.size());
    for (const auto& field : field_infos) {
        field_offsets.emplace_back(foxglove::CreatePackedElementField(builder, builder.CreateString(field.name), field.offset, field.numeric_type));
    }

    const auto fields_vector = builder.CreateVector(field_offsets);
    const auto data_vector = builder.CreateVector(data);
    const auto pose_offset = detail::CreateIdentityPose(builder);

    auto point_cloud_offset =
        foxglove::CreatePointCloud(builder, &timestamp_struct, frame_id_offset, pose_offset, point_stride, fields_vector, data_vector);
    foxglove::FinishPointCloudBuffer(builder, point_cloud_offset);
    return true;
}

/**
 * @brief 将图像转换为Foxglove CompressedImage消息
 * @param image 图像数据
 * @param frame_id 坐标系名称
 * @param format 压缩格式（jpeg/png/webp）
 * @param quality 压缩质量
 * @param builder FlatBuffer构造器
 * @return 成功返回true
 */
inline bool BuildFoxgloveCompressedImage(
    const slam_core::Image& image,
    std::string_view frame_id,
    std::string_view format,
    int quality,
    flatbuffers::FlatBufferBuilder& builder)
{
    builder.Clear();

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

    const foxglove::Time timestamp_struct = detail::MakeFoxgloveTime(image.timestamp());
    const auto frame_id_offset = builder.CreateString(frame_id.data(), frame_id.size());
    const auto format_offset = builder.CreateString(format.data(), format.size());
    const auto data_offset = builder.CreateVector(compressed);

    const auto compressed_image_offset = foxglove::CreateCompressedImage(builder, &timestamp_struct, frame_id_offset, data_offset, format_offset);
    foxglove::FinishCompressedImageBuffer(builder, compressed_image_offset);
    return true;
}

}  // namespace ms_slam::slam_adapter
