#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <foxglove/channel.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/server.hpp>
#include <iox2/iceoryx2.hpp>
#include <mcap/writer.hpp>
#include <spdlog/spdlog.h>

#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

// FlatBuffers generated headers with binary schemas
#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>
#include <fbs/PointCloud_generated.h>

namespace ms_slam::slam_recorder
{

class FoxgloveWebSocketBridge
{
  public:
    struct WebSocketConfig
    {
        bool enable = true;
        std::string host = "127.0.0.1";
        uint16_t port = 8765;
        std::string server_name = "slam_recorder";
        uint32_t poll_interval_ms = 10;
    };

    struct TopicConfig
    {
        std::string name;
        std::string schema;
        bool enabled = true;
    };

    struct RecorderConfig
    {
        bool enable = true;
        std::string output_dir = "./output";
        std::string filename_prefix = "recording";
        bool auto_start = true;
        std::string compression = "zstd";  // none, lz4, zstd
        uint64_t chunk_size = 1048576;     // 1MB
    };

    struct Config
    {
        WebSocketConfig websocket;
        RecorderConfig recorder;
        std::vector<TopicConfig> topics;  // 共享的 topic 配置
    };

    explicit FoxgloveWebSocketBridge(const Config& config);
    ~FoxgloveWebSocketBridge();

    // 禁用拷贝
    FoxgloveWebSocketBridge(const FoxgloveWebSocketBridge&) = delete;
    FoxgloveWebSocketBridge& operator=(const FoxgloveWebSocketBridge&) = delete;

    /// 启动服务（非阻塞）
    void start();

    /// 停止所有服务（阻塞，等待线程退出）
    void stop();

    /// 检查服务是否正在运行
    bool is_running() const { return running_.load(); }

    /// 录制控制
    void start_recording();
    void stop_recording();
    bool is_recording() const { return recording_.load(); }

    /// 获取统计信息
    struct TopicStatistics
    {
        uint64_t forwarded = 0;
        uint64_t recorded = 0;
        uint64_t errors = 0;
    };

    struct Statistics
    {
        std::map<std::string, TopicStatistics> topics;  // topic_name -> stats

        // 兼容性：总计数
        uint64_t total_forwarded() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.forwarded; }
            return total;
        }
        uint64_t total_recorded() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.recorded; }
            return total;
        }
        uint64_t total_errors() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.errors; }
            return total;
        }
    };
    Statistics get_statistics() const;

  private:
    /// 主运行循环（在独立线程中执行）
    void run();

    /// 初始化 MCAP 录制器
    void init_mcap_writer();

    /// 关闭 MCAP 录制器
    void close_mcap_writer();

    /// 生成输出文件名
    std::string generate_output_filename() const;

    /// 轮询并处理数据（按 topic 名称）
    void poll_and_forward_topic(const std::string& topic_name, const std::string& schema);

    /// 录制数据到 MCAP
    void record_to_mcap(const std::string& topic_name, const std::string& schema,
                        const uint8_t* data, size_t size, uint64_t timestamp_ns);

    /// 获取当前时间戳（纳秒）
    static uint64_t get_current_timestamp_ns();

    // 配置
    Config config_;

    // Foxglove Context（必须在 server 和 channels 之间共享）
    foxglove::Context context_;

    // WebSocket 服务器和 channels（topic_name -> channel）
    std::unique_ptr<foxglove::WebSocketServer> server_;
    std::map<std::string, std::unique_ptr<foxglove::RawChannel>> channels_;

    // iceoryx2 node 和订阅者（topic_name -> subscriber）
    // 使用 void* 存储不同类型的 subscriber，实际类型由 schema 决定
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxglovePointCloud>>> pc_subs_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxgloveCompressedImage>>> img_subs_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxgloveImu>>> imu_subs_;

    // MCAP 录制器
    std::unique_ptr<mcap::McapWriter> mcap_writer_;
    std::map<std::string, uint16_t> topic_to_channel_id_;  // topic name -> MCAP channel ID
    uint16_t next_channel_id_ = 1;
    std::string current_output_file_;

    // 线程控制
    std::atomic<bool> running_{false};
    std::atomic<bool> recording_{false};
    std::unique_ptr<std::thread> worker_thread_;

    // 统计信息（topic_name -> stats）
    mutable std::map<std::string, std::atomic<uint64_t>> forwarded_count_;
    mutable std::map<std::string, std::atomic<uint64_t>> recorded_count_;
    mutable std::map<std::string, std::atomic<uint64_t>> error_count_;
};

}  // namespace ms_slam::slam_recorder