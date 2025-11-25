#pragma once

#include <spdlog/spdlog.h>

#include <foxglove/CompressedImage.pb.h>
#include <foxglove/FrameTransforms.pb.h>
#include <foxglove/Imu.pb.h>
#include <foxglove/PointCloud.pb.h>
#include <foxglove/PoseInFrame.pb.h>
#include <foxglove/PosesInFrame.pb.h>
#include <foxglove/SceneUpdate.pb.h>

namespace ms_slam::slam_common
{
/**
 * @brief Foxglove Protobuf 类型别名，统一暴露常用消息名称
 *
 * @note 通过使用别名，原有依赖 FlatBuffers 的调用点可以平滑迁移到 Protobuf。
 */
using FoxglovePointCloud = ::foxglove::PointCloud;
using FoxgloveCompressedImage = ::foxglove::CompressedImage;
using FoxgloveImu = ::foxglove::Imu;
using FoxglovePose = ::foxglove::Pose;
using FoxglovePoseInFrame = ::foxglove::PoseInFrame;
using FoxglovePosesInFrame = ::foxglove::PosesInFrame;
using FoxgloveFrameTransforms = ::foxglove::FrameTransforms;
using FoxgloveSceneUpdate = ::foxglove::SceneUpdate;

}  // namespace ms_slam::slam_common
