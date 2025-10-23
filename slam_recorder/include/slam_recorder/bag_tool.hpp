#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace ms_slam::slam_recorder {

enum class InputType {
    Ros1Mcap,
    FlatbufferMcap,
    Rosbag,
};

struct TopicSettings {
    bool playback{true};
    bool record{true};
    std::string publish_service;
    uint32_t queue_depth{10};
};

struct ToolConfig {
    std::string input_path;
    InputType input_type{InputType::Ros1Mcap};
    bool default_playback{true};
    bool default_record{true};
    bool playback_enabled{false};
    bool playback_sync_time{true};
    double playback_rate{1.0};
    uint32_t playback_queue_depth{10};
    bool record_enabled{false};
    std::string record_output_dir{"./output"};
    std::string record_prefix{"recording"};
    std::string record_compression{"zstd"};
    uint64_t record_chunk_size{1024ull * 1024ull};
    bool record_overwrite{true};
    std::unordered_map<std::string, TopicSettings> topics;
};

ToolConfig load_bag_tool_config(const std::string& yaml_path);

}  // namespace ms_slam::slam_recorder
