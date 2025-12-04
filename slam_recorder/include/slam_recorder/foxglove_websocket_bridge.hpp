/**
 * @file foxglove_websocket_bridge.hpp
 * @brief Foxglove WebSocket 桥接器接口定义
 */

#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/message.h>
#include <google/protobuf/timestamp.pb.h>
#include <foxglove/channel.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/server.hpp>
#include <iox2/iceoryx2.hpp>
#include <mcap/writer.hpp>
#include <spdlog/spdlog.h>
#include <slam_common/iceoryx_pub_sub.hpp>
#include <slam_common/sensor_struct.hpp>

#include "foxglove/CompressedImage.pb.h"
#include "foxglove/FrameTransforms.pb.h"
#include "foxglove/Imu.pb.h"
#include "foxglove/PackedElementField.pb.h"
#include "foxglove/PointCloud.pb.h"
#include "foxglove/PoseInFrame.pb.h"
#include "foxglove/PosesInFrame.pb.h"
#include "foxglove/SceneUpdate.pb.h"

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
    struct WebSocketConfig {
        bool enable = true;                         ///< 是否启用 WebSocket 服务
        std::string host = "127.0.0.1";             ///< WebSocket 监听地址
        uint16_t port = 8765;                       ///< WebSocket 端口
        std::string server_name = "slam_recorder";  ///< 服务名称
        uint32_t poll_interval_ms = 10;             ///< 轮询周期（毫秒）
    };

    /**
     * @brief Topic 配置
     */
    struct TopicConfig {
        std::string name;     ///< Topic 名称
        std::string schema;   ///< Protobuf Schema 名称
        bool enabled = true;  ///< 是否启用
    };

    /**
     * @brief MCAP 录制器配置
     */
    struct RecorderConfig {
        bool enable = true;                         ///< 是否启用录制
        std::string output_dir = "./output";        ///< 输出目录
        std::string filename_prefix = "recording";  ///< 文件名前缀
        bool auto_start = true;                     ///< 是否自动开始录制
        std::string compression = "zstd";           ///< 压缩方式（none/lz4/zstd）
        uint64_t chunk_size = 1048576;              ///< Chunk 大小（字节）
    };

    /**
     * @brief 桥接器总配置
     */
    struct Config {
        WebSocketConfig websocket;        ///< WebSocket 配置
        RecorderConfig recorder;          ///< MCAP 录制配置
        std::vector<TopicConfig> topics;  ///< Topic 列表
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
    void Start();

    /**
     * @brief 停止桥接器并等待线程退出
     */
    void Stop();

    /**
     * @brief 检查桥接器是否正在运行
     * @return 运行中返回 true
     */
    bool IsRunning() const { return running_.load(); }

    /**
     * @brief 启动 MCAP 录制
     */
    void StartRecording();

    /**
     * @brief 停止 MCAP 录制
     */
    void StopRecording();

    /**
     * @brief 判断是否正在录制
     * @return 正在录制返回 true
     */
    bool IsRecording() const { return recording_.load(); }

    /**
     * @brief Topic 统计数据
     */
    struct TopicStatistics {
        uint64_t forwarded = 0;  ///< 转发次数
        uint64_t recorded = 0;   ///< 录制次数
        uint64_t errors = 0;     ///< 错误次数
    };

    /**
     * @brief 全局统计数据
     */
    struct Statistics {
        std::map<std::string, TopicStatistics> topics;  ///< Topic 统计映射

        /**
         * @brief 计算转发总次数
         * @return 转发总数
         */
        uint64_t total_forwarded() const
        {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) {
                total += stats.forwarded;
            }
            return total;
        }

        /**
         * @brief 计算录制总次数
         * @return 录制总数
         */
        uint64_t total_recorded() const
        {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) {
                total += stats.recorded;
            }
            return total;
        }

        /**
         * @brief 计算错误总次数
         * @return 错误总数
         */
        uint64_t total_errors() const
        {
            uint64_t total = 0;
            for (const auto& [_, stats] : topics) {
                total += stats.errors;
            }
            return total;
        }
    };

    /**
     * @brief 获取当前统计信息
     * @return 统计数据副本
     */
    Statistics GetStatistics() const;

  private:
    /**
     * @brief 主运行循环（在独立线程中执行）
     */
    void Run();

    /**
     * @brief 初始化 MCAP 录制器
     */
    void InitMcapWriter();

    /**
     * @brief 关闭 MCAP 录制器
     */
    void CloseMcapWriter();

    /**
     * @brief 生成输出文件名
     * @return 输出文件路径
     */
    std::string GenerateOutputFilename() const;

    /**
     * @brief 轮询并处理指定 Topic
     * @param topic_name Topic 名称
     * @param schema Schema 名称
     */
    void PollAndForwardTopic(const std::string& topic_name, const std::string& schema);

    /**
     * @brief 将数据写入 MCAP 文件
     * @param topic_name Topic 名称
     * @param schema Schema 名称
     * @param data 数据缓冲
     * @param timestamp_ns 时间戳（纳秒）
     */
    void RecordToMcap(const std::string& topic_name, const std::string& schema, const std::string& data, uint64_t timestamp_ns);

    /**
     * @brief 获取当前时间戳（纳秒）
     * @return 当前时间戳
     */
    static uint64_t GetCurrentTimestampNs();

    /**
     * @brief 将消息时间对齐到系统时间轴
     * @param message_time_ns 消息自身时间（纳秒）
     * @return 对齐后的纳秒时间戳
     */
    uint64_t AlignTimestamp(uint64_t message_time_ns);

    /**
     * @brief 确保全局时间戳严格递增（跨所有通道）
     * @param timestamp_ns 当前时间戳
     * @return 递增后的时间戳
     */
    uint64_t EnsureGlobalMonotonic(uint64_t timestamp_ns);

    /**
     * @brief 为 Topic 创建 WebSocket schema
     * @param schema_name Schema 名称
     * @return 构造好的 Schema
     */
    foxglove::Schema BuildWsSchema(const std::string& schema_name);

    /**
     * @brief 基于 Protobuf 描述符构建 MCAP / WebSocket 描述符数据
     * @param descriptor 描述符指针
     * @return 序列化后的描述符集
     */
    static std::string BuildDescriptorSet(const google::protobuf::Descriptor* descriptor);

    /**
     * @brief 将 google::protobuf::Timestamp 转为纳秒
     * @param stamp 时间戳
     * @return 纳秒
     */
    static uint64_t ToNanoseconds(const google::protobuf::Timestamp& stamp);

    /**
     * @brief 序列化 Protobuf 消息
     * @param message 输入消息
     * @param buffer 输出缓冲
     * @return 成功返回 true
     */
    static bool SerializeMessage(const google::protobuf::Message& message, std::string& buffer);

    /**
     * @brief 注册 iceoryx2 订阅与回调
     * @tparam MessageType 消息类型
     * @param topic_name 话题名称
     * @param schema_name Schema 名称
     * @return 创建好的订阅者
     */
    template <typename MessageType>
    std::shared_ptr<slam_common::IoxSubscriber<MessageType>> RegisterSubscriber(
        const std::string& topic_name,
        const std::string& schema_name,
        const std::function<bool(const MessageType&, std::string&, uint64_t&)>& converter,
        const slam_common::IoxPubSubConfig& iox_config = slam_common::IoxPubSubConfig());

    // 配置
    Config config_;

    // Foxglove Context（必须在 server 和 channels 之间共享）
    foxglove::Context context_;

    // WebSocket 服务器和 channels（topic_name -> channel）
    std::unique_ptr<foxglove::WebSocketServer> server_;
    std::map<std::string, std::unique_ptr<foxglove::RawChannel>> channels_;

    // iceoryx2 节点
    std::shared_ptr<slam_common::IoxNode> iox_node_;

    // iceoryx2 订阅者（topic_name -> subscriber）
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::LivoxPointCloudDate>>> pc_subs_;
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::ImageDate>>> img_subs_;
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::LivoxImuData>>> imu_subs_;
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::OdomData>>> pose_subs_;
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::PathData>>> poses_subs_;
    std::unordered_map<std::string, std::shared_ptr<slam_common::IoxSubscriber<slam_common::FrameTransformArray>>> frame_tf_subs_;

    struct PendingPacket {
        std::string data;
        uint64_t timestamp_ns{0};
    };

    // 待转发缓存与互斥
    std::unordered_map<std::string, std::vector<PendingPacket>> pending_packets_;
    std::mutex pending_mutex_;

    // MCAP 录制器
    std::unique_ptr<mcap::McapWriter> mcap_writer_;
    std::map<std::string, uint16_t> topic_to_channel_id_;  ///< Topic 名称到 MCAP channel ID
    uint16_t next_channel_id_ = 1;
    std::string current_output_file_;
    std::unordered_map<std::string, std::string> schema_descriptor_cache_;
    std::unordered_map<std::string, mcap::SchemaId> mcap_schema_cache_;

    std::atomic<uint64_t> last_message_time_ns_{0};
    std::atomic<uint64_t> time_offset_ns_{0};
    std::atomic<bool> time_sync_initialized_{false};
    std::atomic<uint64_t> last_global_timestamp_ns_{0};

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
