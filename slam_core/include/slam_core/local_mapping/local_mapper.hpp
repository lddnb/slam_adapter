#pragma once

#include <memory>
#include <optional>

#include <spdlog/spdlog.h>

#include "slam_core/local_mapping/local_mapping_types.hpp"

namespace ms_slam::slam_core::local_mapping
{
/**
 * @brief 局部建图接口，消费里程计输出并维护独立 voxel 地图
 */
class LocalMapper
{
  public:
    /**
     * @brief 虚析构，便于多态释放
     */
    virtual ~LocalMapper() = default;

    /**
     * @brief 推入一帧里程计输出，内部缓存等待建图线程处理
     * @param input 里程计输出
     */
    virtual void PushOdometryOutput(const OdometryOutput& input) = 0;

    /**
     * @brief 驱动一次局部建图流程（关键帧判定、因子构建、滑窗优化）
     * @return 若完成优化则返回结果，否则为空
     */
    virtual std::optional<LocalMappingResult> TryProcess() = 0;

    /**
     * @brief 导出局部建图优化后的点云地图
     * 
     * @param out 
     */
    virtual void ExportMapCloud(std::vector<PointCloudType::Ptr>& out) = 0;

    /**
     * @brief 导出局部建图优化后的状态量
     * 
     * @param out 
     */
    virtual void ExportStates(std::unordered_map<int, CommonState>& out) = 0;

    /**
     * @brief 重新加载关键帧地图或清空内部状态
     * @return void
     */
    virtual void Reset() = 0;
};

/**
 * @brief 创建基于 VoxelSLAM 求解器的局部建图实例
 * @param config 局部建图配置
 * @return 局部建图指针
 */
std::unique_ptr<LocalMapper> CreateVoxelLocalMapper(const LocalMapperConfig& config);

}  // namespace ms_slam::slam_core::local_mapping
