#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace ms_slam::slam_recorder
{

/**
 * @brief 输入数据源类型
 * @note 该枚举用于指示 bag_tool 的输入格式，以便后续选择对应的解析路径
 */
enum class InputType {
    Ros1Mcap,
    Rosbag,
};

/**
 * @brief 单个话题的回放配置
 */
struct TopicSettings {
    bool playback{true};          ///< 是否启用回放
    std::string publish_service;  ///< 回放时映射的服务名称
    std::string schema;           ///< 话题的消息 Schema 名称（用于预创建发布器）
};

/**
 * @brief bag_tool 的整体运行配置
 */
struct ToolConfig {
    std::string input_path;                                 ///< 输入数据文件路径
    InputType input_type{InputType::Ros1Mcap};              ///< 输入数据类型
    bool default_playback{true};                            ///< 全局默认回放开关
    double processing_start_seconds{0.0};                   ///< 处理窗口起始时间（秒）
    double processing_duration_seconds{0.0};                ///< 处理窗口持续时间（秒）
    bool playback_enabled{false};                           ///< 是否启用回放功能
    bool playback_sync_time{true};                          ///< 回放是否同步时间
    double playback_rate{1.0};                              ///< 回放倍率
    std::unordered_map<std::string, TopicSettings> topics;  ///< 话题级别的覆盖配置
};

/**
 * @brief 从 YAML 配置文件加载 bag_tool 设置
 * @param yaml_path 配置文件绝对路径或相对路径
 * @return 解析后的工具配置
 * @note 解析失败时会抛出 std::runtime_error 异常
 */
ToolConfig LoadBagToolConfig(const std::string& yaml_path);

}  // namespace ms_slam::slam_recorder
