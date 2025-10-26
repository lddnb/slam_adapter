/**
 * @file foxglove_websocket_bridge.hpp
 * @brief Foxglove WebSocket 桥接器接口定义
 */

#pragma once

#include <atomic>
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
#include <fbs/CompressedImage_generated.h>
#include <fbs/Imu_generated.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/PoseInFrame_generated.h>
#include <fbs/PosesInFrame_generated.h>
#include <fbs/FrameTransforms_generated.h>
#include <fbs/SceneUpdate_generated.h>

namespace ms_slam::slam_recorder
{

/**
 * @brief Foxglove WebSocket 桥接器，负责从 iceoryx2 获取数据并转发至 Foxglove 与 MCAP
 */
class FoxgloveWebSocketBridge
{
  public:
    /**
     * @brief WebSocket 服务配置
     */
    struct WebSocketConfig
    {
        bool enable = true;                ///< 是否启用 WebSocket 服务
        std::string host = "127.0.0.1";   ///< WebSocket 监听地址
        uint16_t port = 8765;              ///< WebSocket 端口
        std::string server_name = "slam_recorder"; ///< 服务名称
        uint32_t poll_interval_ms = 10;    ///< 轮询周期（毫秒）
    };

    /**
     * @brief Topic 配置
     */
    struct TopicConfig
    {
        std::string name;    ///< Topic 名称
        std::string schema;  ///< FlatBuffers Schema 名称
        bool enabled = true; ///< 是否启用
    };

    /**
     * @brief MCAP 录制器配置
     */
    struct RecorderConfig
    {
        bool enable = true;                    ///< 是否启用录制
        std::string output_dir = "./output";  ///< 输出目录
        std::string filename_prefix = "recording"; ///< 文件名前缀
        bool auto_start = true;                 ///< 是否自动开始录制
        std::string compression = "zstd";      ///< 压缩方式（none/lz4/zstd）
        uint64_t chunk_size = 1048576;          ///< Chunk 大小（字节）
    };

    /**
     * @brief 桥接器总配置
     */
    struct Config
    {
        WebSocketConfig websocket;                ///< WebSocket 配置
        RecorderConfig recorder;                  ///< MCAP 录制配置
        std::vector<TopicConfig> topics;          ///< Topic 列表
    };

    /**
     * @brief 构造 Foxglove WebSocket 桥接器
     * @param config 桥接器配置
     */
    explicit FoxgloveWebSocketBridge(const Config& config);

    /**
     * @brief 析构函数，自动停止所有资源
     */
    ~FoxgloveWebSocketBridge();

    // 禁用拷贝
    FoxgloveWebSocketBridge(const FoxgloveWebSocketBridge&) = delete;
    FoxgloveWebSocketBridge& operator=(const FoxgloveWebSocketBridge&) = delete;

    /**
     * @brief 启动桥接器（非阻塞）
     */
    void start();

    /**
     * @brief 停止桥接器并等待线程退出
     */
    void stop();

    /**
     * @brief 检查桥接器是否正在运行
     * @return 运行中返回 true
     */
    bool is_running() const { return running_.load(); }

    /**
     * @brief 启动 MCAP 录制
     */
    void start_recording();

    /**
     * @brief 停止 MCAP 录制
     */
    void stop_recording();

    /**
     * @brief 判断是否正在录制
     * @return 正在录制返回 true
     */
    bool is_recording() const { return recording_.load(); }

    /**
     * @brief Topic 统计数据
     */
    struct TopicStatistics
    {
        uint64_t forwarded = 0; ///< 转发次数
        uint64_t recorded = 0;  ///< 录制次数
        uint64_t errors = 0;    ///< 错误次数
    };

    /**
     * @brief 全局统计数据
     */
    struct Statistics
    {
        std::map<std::string, TopicStatistics> topics;  ///< Topic 统计映射

        /**
         * @brief 计算转发总次数
         * @return 转发总数
         */
        uint64_t total_forwarded() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.forwarded; }
            return total;
        }

        /**
         * @brief 计算录制总次数
         * @return 录制总数
         */
        uint64_t total_recorded() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.recorded; }
            return total;
        }

        /**
         * @brief 计算错误总次数
         * @return 错误总数
         */
        uint64_t total_errors() const {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) { total += stats.errors; }
            return total;
        }
    };

    /**
     * @brief 获取当前统计信息
     * @return 统计数据副本
     */
    Statistics get_statistics() const;

  private:
    /**
     * @brief 主运行循环（在独立线程中执行）
     */
    void run();

    /**
     * @brief 初始化 MCAP 录制器
     */
    void init_mcap_writer();

    /**
     * @brief 关闭 MCAP 录制器
     */
    void close_mcap_writer();

    /**
     * @brief 生成输出文件名
     * @return 输出文件路径
     */
    std::string generate_output_filename() const;

    /**
     * @brief 轮询并处理指定 Topic
     * @param topic_name Topic 名称
     * @param schema Schema 名称
     */
    void poll_and_forward_topic(const std::string& topic_name, const std::string& schema);

    /**
     * @brief 将数据写入 MCAP 文件
     * @param topic_name Topic 名称
     * @param schema Schema 名称
     * @param data 数据指针
     * @param size 数据长度
     * @param timestamp_ns 时间戳（纳秒）
     */
    void record_to_mcap(const std::string& topic_name, const std::string& schema,
                        const uint8_t* data, size_t size, uint64_t timestamp_ns);

    /**
     * @brief 获取当前时间戳（纳秒）
     * @return 当前时间戳
     */
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
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxglovePoseInFrame>>> pose_subs_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxglovePosesInFrame>>> poses_subs_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxgloveFrameTransforms>>> frame_tf_subs_;
    std::map<std::string, std::unique_ptr<slam_common::FBSSubscriber<slam_common::FoxgloveSceneUpdate>>> frame_marker_subs_;

    // MCAP 录制器
    std::unique_ptr<mcap::McapWriter> mcap_writer_;
    std::map<std::string, uint16_t> topic_to_channel_id_;  ///< Topic 名称到 MCAP channel ID
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
