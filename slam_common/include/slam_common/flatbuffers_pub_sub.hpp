/**
 * @file flatbuffers_pub_sub.hpp
 * @brief 基于 FlatBuffers 的发布订阅封装
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>
#include <thread>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <iox2/iceoryx2.hpp>
#include <spdlog/spdlog.h>

namespace ms_slam::slam_common
{
/**
 * @brief 发布订阅配置
 */
struct PubSubConfig {
    /// 最大发布者数量
    uint32_t max_publishers{1};
    /// 最大订阅者数量
    uint32_t max_subscribers{3};
    /// 初始最大共享内存切片大小（默认 16MB）
    uint64_t initial_max_slice_len{16 * 1024 * 1024};
    /// 分配策略配置
    iox2::AllocationStrategy allocation_strategy{iox2::AllocationStrategy::Static};
    /// 单个订阅者缓冲队列深度
    uint32_t subscriber_max_buffer_size{10};
    /// 轮询间隔
    std::chrono::milliseconds poll_interval{3};
};

/**
 * @brief 通用 FlatBuffers 发布器（事件驱动）
 * @tparam MessageType 业务消息类型
 */
template <typename MessageType>
class FBSPublisher
{
  public:
    /// 发布结果回调
    using PublishCallback = std::function<void(uint32_t seq, const MessageType&)>;

    /**
     * @brief 构造发布器并初始化 iceoryx 服务
     * @param node iceoryx 节点
     * @param service_name 服务名
     * @param config 发布订阅配置
     */
    FBSPublisher(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
                 const std::string& service_name,
                 PubSubConfig config = PubSubConfig())
    : node_(std::move(node))
    , service_name_(service_name)
    , config_(config)
    , published_count_(0)
    {
        auto iox_service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");

        service_ = node_->service_builder(iox_service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .subscriber_max_buffer_size(config_.subscriber_max_buffer_size)
                       .enable_safe_overflow(true)
                       .open_or_create()
                       .expect("successful service open/create");

        publisher_ = service_->publisher_builder()
                         .initial_max_slice_len(config_.initial_max_slice_len)
                         .allocation_strategy(config_.allocation_strategy)
                         .create()
                         .expect("successful publisher creation");

        spdlog::info(
            "FlatBuffers Publisher service '{}' initialized with {} max subscribers, {} max buffer size, {}MB initial max slice len",
            service_name_,
            config_.max_subscribers,
            config_.subscriber_max_buffer_size,
            config_.initial_max_slice_len / 1024 / 1024);
    }

    /**
     * @brief 设置发布成功回调
     * @param callback 回调函数
     */
    void set_publish_callback(PublishCallback callback)
    {
        publish_callback_ = callback;
    }

    /**
     * @brief 发布一条业务消息
     * @param message 待发布消息
     * @return 发布成功返回 true
     */
    bool publish(const MessageType& message)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffers Publisher service not initialized!");
        }

        try {
            auto buffer = message.serialize();
            auto sample = publisher_->loan_slice_uninit(buffer.size()).expect("acquire sample");
            auto initialized_sample = sample.write_from_fn([&buffer](uint64_t idx) { return buffer.data()[idx]; });
            iox2::send(std::move(initialized_sample)).expect("send successful");

            published_count_++;

            if (publish_callback_) {
                publish_callback_(published_count_.load(), message);
            }

            return true;
        } catch (const std::exception& e) {
            spdlog::error("FlatBuffers publish error: {}", e.what());
            return false;
        }
    }

    /**
     * @brief 直接发布已序列化的字节流（零拷贝）
     * @param data 序列化数据指针
     * @param size 数据长度（字节）
     * @return 发布成功返回 true
     */
    bool publish_raw(const uint8_t* data, size_t size)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffers Publisher service not initialized!");
        }

        try {
            auto sample = publisher_->loan_slice_uninit(size).expect("acquire sample");
            auto initialized_sample = sample.write_from_fn([data](uint64_t idx) { return data[idx]; });
            iox2::send(std::move(initialized_sample)).expect("send successful");
            published_count_++;

            return true;
        } catch (const std::exception& e) {
            spdlog::error("FlatBuffers publish_raw error: {}", e.what());
            return false;
        }
    }

    /**
     * @brief 从 FlatBufferBuilder 发布（零拷贝）
     * @param fbb FlatBufferBuilder 对象
     * @return 发布成功返回 true
     */
    bool publish_from_builder(flatbuffers::FlatBufferBuilder& fbb)
    {
        return publish_raw(fbb.GetBufferPointer(), fbb.GetSize());
    }

    /**
     * @brief 获取累计发布数量
     * @return 已发布消息计数
     */
    uint32_t get_published_count() const { return published_count_.load(); }

    /**
     * @brief 检查发布器是否已就绪
     * @return 就绪返回 true
     */
    bool is_ready() const { return publisher_.has_value(); }

  private:
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    PubSubConfig config_;
    std::atomic<uint32_t> published_count_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Publisher<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> publisher_;

    PublishCallback publish_callback_;
};

/**
 * @brief 通用 FlatBuffers 订阅器（按需拉取）
 * @tparam MessageType 业务消息类型
 */
template <typename MessageType>
class FBSSubscriber
{
  public:
    /// 接收回调
    using ReceiveCallback = std::function<void(const MessageType&)>;

    /**
     * @brief 构造订阅器
     * @param node iceoryx 节点
     * @param service_name 服务名
     * @param receive_callback 接收回调
     * @param config 发布订阅配置
     */
    FBSSubscriber(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
                  const std::string& service_name,
                  ReceiveCallback receive_callback = nullptr,
                  PubSubConfig config = PubSubConfig())
    : node_(node)
    , service_name_(service_name)
    , receive_callback_(receive_callback)
    , config_(config)
    , received_count_(0)
    {
        auto iox_service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");

        service_ = node_->service_builder(iox_service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .subscriber_max_buffer_size(config_.subscriber_max_buffer_size)
                       .enable_safe_overflow(true)
                       .open_or_create()
                       .expect("successful service open/create");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");
        spdlog::info(
            "FlatBuffers Subscriber service '{}' initialized with {} max subscribers, {} max buffer size",
            service_name_,
            config_.max_subscribers,
            config_.subscriber_max_buffer_size);
    }

    /**
     * @brief 设置接收回调
     * @param callback 回调函数
     */
    void set_receive_callback(ReceiveCallback callback)
    {
        receive_callback_ = callback;
    }

    /**
     * @brief 接收单条消息
     * @return 有效消息则返回 optional
     */
    iox::optional<MessageType> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                auto message = MessageType::deserialize(data, size);

                if (receive_callback_) {
                    receive_callback_(message);
                }

                return message;
            }
        } catch (const std::exception& e) {
            spdlog::error("FlatBuffers receive error: {}", e.what());
        }
        return iox::nullopt;
    }

    /**
     * @brief 接收所有可用消息
     * @return 消息列表
     */
    std::vector<MessageType> receive_all()
    {
        std::vector<MessageType> messages;
        auto message = receive_once();
        while (message.has_value()) {
            messages.push_back(std::move(message.value()));
            message = receive_once();
        }
        return messages;
    }

    /**
     * @brief 接收单条原始字节流（带拷贝）
     * @return 有效数据则返回字节数组
     */
    iox::optional<std::vector<uint8_t>> receive_raw_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                return std::vector<uint8_t>(data, data + size);
            }
        } catch (const std::exception& e) {
            spdlog::error("FlatBuffers receive_raw error: {}", e.what());
        }
        return iox::nullopt;
    }

    /**
     * @brief 批量接收所有原始字节流（带拷贝）
     * @return 原始字节数组列表
     */
    std::vector<std::vector<uint8_t>> receive_all_raw()
    {
        std::vector<std::vector<uint8_t>> raw_messages;

        while (true) {
            try {
                auto sample = subscriber_->receive().expect("receive succeeds");
                if (!sample.has_value()) {
                    break;
                }

                received_count_++;

                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                raw_messages.emplace_back(data, data + size);

            } catch (const std::exception& e) {
                spdlog::error("FlatBuffers receive_all_raw error: {}", e.what());
                break;
            }
        }

        return raw_messages;
    }

    /**
     * @brief 检查是否有待处理数据
     * @return 有数据返回 true
     */
    bool has_data() const
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            return sample.has_value();
        } catch (const std::exception&) {
            return false;
        }
    }

    /**
     * @brief 获取累计接收数量
     * @return 已接收消息数量
     */
    uint32_t get_received_count() const { return received_count_.load(); }

    /**
     * @brief 检查订阅者是否已就绪
     * @return 就绪返回 true
     */
    bool is_ready() const { return subscriber_.has_value(); }

  private:
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    PubSubConfig config_;
    std::atomic<uint32_t> received_count_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> subscriber_;
};

/**
 * @brief 线程化 FlatBuffers 订阅器
 * @tparam MessageType 业务消息类型
 */
template <typename MessageType>
class ThreadedFBSSubscriber
{
  public:
    /// 接收回调
    using ReceiveCallback = std::function<void(const MessageType&)>;

    /**
     * @brief 构造线程化订阅器
     * @param node iceoryx 节点
     * @param service_name 服务名
     * @param receive_callback 接收回调
     * @param config 发布订阅配置
     */
    ThreadedFBSSubscriber(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
                          const std::string& service_name,
                          ReceiveCallback receive_callback,
                          PubSubConfig config = PubSubConfig())
    : node_(node)
    , service_name_(service_name)
    , receive_callback_(receive_callback)
    , config_(config)
    , received_count_(0)
    , running_(false)
    , should_stop_(false)
    {
        auto iox_service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");

        service_ = node_->service_builder(iox_service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .subscriber_max_buffer_size(config_.subscriber_max_buffer_size)
                       .enable_safe_overflow(true)
                       .open_or_create()
                       .expect("successful service open/create");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");

        spdlog::info(
            "Threaded FlatBuffers Subscriber service '{}' initialized with {} max subscribers, {} max buffer size",
            service_name_,
            config_.max_subscribers,
            config_.subscriber_max_buffer_size);
    }

    /**
     * @brief 析构函数，确保线程停止
     */
    ~ThreadedFBSSubscriber()
    {
        stop();
    }

    /**
     * @brief 启动后台轮询线程
     */
    void start()
    {
        if (!running_.load()) {
            should_stop_.store(false);
            thread_ = std::thread(&ThreadedFBSSubscriber::run, this);
            running_.store(true);
        }
    }

    /**
     * @brief 停止后台轮询线程
     */
    void stop()
    {
        if (running_.load()) {
            should_stop_.store(true);
            if (thread_.joinable()) {
                thread_.join();
            }
            running_.store(false);
        }
    }

    /**
     * @brief 判断线程是否正在运行
     * @return 运行中返回 true
     */
    bool is_running() const { return running_.load(); }

    /**
     * @brief 更新接收回调
     * @param callback 新回调
     */
    void set_receive_callback(ReceiveCallback callback)
    {
        receive_callback_ = callback;
    }

    /**
     * @brief 获取累计接收数量
     * @return 已接收消息数量
     */
    uint32_t get_received_count() const { return received_count_.load(); }

    /**
     * @brief 检查订阅者是否已就绪
     * @return 就绪返回 true
     */
    bool is_ready() const { return subscriber_.has_value(); }

  private:
    void run()
    {
        while (!should_stop_.load()) {
            try {
                receive_messages();
                std::this_thread::sleep_for(config_.poll_interval);
            } catch (const std::exception& e) {
                spdlog::error("ThreadedFlatBuffer Subscriber error: {}", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void receive_messages()
    {
        auto sample = subscriber_->receive().expect("receive succeeds");
        while (sample.has_value()) {
            try {
                received_count_++;

                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                auto message = MessageType::deserialize(data, size);

                if (receive_callback_) {
                    receive_callback_(message);
                }
            } catch (const std::exception& e) {
                spdlog::error("Message processing error: {}", e.what());
            }

            sample = subscriber_->receive().expect("receive succeeds");
        }
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    PubSubConfig config_;
    std::atomic<uint32_t> received_count_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> subscriber_;

    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::thread thread_;
};

}  // namespace ms_slam::slam_common
