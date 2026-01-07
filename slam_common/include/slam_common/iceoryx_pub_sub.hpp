#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <iox2/iceoryx2.hpp>

#include "slam_common/sensor_struct.hpp"

namespace ms_slam::slam_common
{
/**
 * @brief iceoryx2 发布订阅配置（定长数据版本）
 */
struct IoxPubSubConfig {
    uint32_t max_publishers{1};               ///< 发布者数量上限
    uint32_t max_subscribers{3};              ///< 订阅者数量上限
    uint32_t subscriber_max_buffer_size{10};  ///< 单订阅者缓冲区深度
    uint32_t history_size{1};                 ///< 订阅者历史缓存深度
    bool enable_safe_overflow{true};          ///< 是否启用安全溢出策略
};

/// 复用的 iceoryx2 节点类型
using IoxNode = iox2::Node<iox2::ServiceType::Ipc>;

/**
 * @brief 基于 iceoryx2 的定长发布器，避免 Slice 序列化
 * @tparam MessageType 传输数据类型，需为平凡可拷贝
 */
template <typename MessageType>
class IoxPublisher
{
  public:
    /// 发布构建回调签名，直接在共享内存中构造消息
    using BuildCallback = std::function<void(MessageType&)>;
    /// 发布构建回调（可返回成功标志），用于一次性发布不复用回调场景
    using BuildResultCallback = std::function<bool(MessageType&)>;

    /**
     * @brief 构造发布器并创建/打开服务
     * @param node 共享的 iceoryx2 节点
     * @param service_name 服务名称，形如 /livox/lidar
     * @param build_callback 共享内存构建回调，可为空
     * @param config 发布订阅配置
     * @return 无
     */
    IoxPublisher(std::shared_ptr<IoxNode> node, std::string service_name, BuildCallback build_callback = nullptr, IoxPubSubConfig config = {})
    : node_(std::move(node)),
      service_name_(std::move(service_name)),
      build_callback_(std::move(build_callback)),
      config_(config),
      published_count_(0)
    {
        static_assert(std::is_trivially_copyable_v<MessageType>, "MessageType must be trivially copyable for zero-copy transport");

        if (!node_) {
            spdlog::error("IoxPublisher requires a valid node, service={}", service_name_);
            return;
        }

        auto service_name_result = iox2::ServiceName::create(service_name_.c_str());
        if (!service_name_result.has_value()) {
            spdlog::error("Invalid iceoryx2 service name '{}', error={}", service_name_, static_cast<int>(service_name_result.error()));
            return;
        }
        auto iox_service_name = std::move(service_name_result).value();

        // 配置服务，全部采用定长 Payload，关闭 Slice 依赖
        auto service_result = node_->service_builder(iox_service_name)
                                  .publish_subscribe<MessageType>()
                                  .max_publishers(config_.max_publishers)
                                  .max_subscribers(config_.max_subscribers)
                                  .subscriber_max_buffer_size(config_.subscriber_max_buffer_size)
                                  .history_size(config_.history_size)
                                  .enable_safe_overflow(config_.enable_safe_overflow)
                                  .open_or_create();

        if (!service_result.has_value()) {
            spdlog::error("Failed to open_or_create iceoryx2 service '{}', error={}", service_name_, static_cast<int>(service_result.error()));
            return;
        }
        service_.emplace(std::move(service_result).value());

        auto publisher_result = service_->publisher_builder().create();
        if (!publisher_result.has_value()) {
            spdlog::error(
                "Failed to create iceoryx2 publisher for service '{}', error={}",
                service_name_,
                static_cast<int>(publisher_result.error()));
            return;
        }
        publisher_.emplace(std::move(publisher_result).value());

        spdlog::info(
            "IoxPublisher ready: service={}, max_pub={}, max_sub={}, buffer={}",
            service_name_,
            config_.max_publishers,
            config_.max_subscribers,
            config_.subscriber_max_buffer_size);
    }

    /**
     * @brief 发布单条消息，基于构建回调直接在共享内存中填充数据
     * @param 无
     * @return 发布成功返回 true
     */
    bool Publish()
    {
        if (!publisher_.has_value()) {
            spdlog::error("IoxPublisher not initialized for service {}", service_name_);
            return false;
        }
        if (!build_callback_) {
            spdlog::error("IoxPublisher build callback is null for service {}", service_name_);
            return false;
        }

        // 通过 loan_uninit 获取未初始化的共享内存块，并原地构造 payload
        auto loan_result = publisher_->loan_uninit();
        if (!loan_result.has_value()) {
            spdlog::error("IoxPublisher failed to loan shared memory for service {}, error={}", service_name_, static_cast<int>(loan_result.error()));
            return false;
        }
        auto sample_uninit = std::move(loan_result).value();
        MessageType& payload = sample_uninit.payload_mut();  // 共享内存中的可写引用
        build_callback_(payload);                            // 原地填充数据

        auto sample_init = iox2::assume_init(std::move(sample_uninit));
        auto send_result = iox2::send(std::move(sample_init));
        if (!send_result.has_value()) {
            spdlog::error("IoxPublisher failed to send sample for service {}, error={}", service_name_, static_cast<int>(send_result.error()));
            return false;
        }

        ++published_count_;
        return true;
    }

    /**
     * @brief 立即执行构建回调并按返回值决定是否发布，适合一次性零拷贝发布，适用于多线程发布保证线程安全
     *        不会有后设的回调可能覆盖前一个发布的构建逻辑，导致数据错乱或“用旧回调发布”
     * @param builder 构建回调，返回 true 表示构建成功、允许发送
     * @return 发布成功返回 true，构建失败或未发送返回 false
     */
    bool PublishWithBuilder(BuildResultCallback builder)
    {
        if (!publisher_.has_value()) {
            spdlog::error("IoxPublisher not initialized for service {}", service_name_);
            return false;
        }
        if (!builder) {
            spdlog::error("IoxPublisher build callback is null for service {}", service_name_);
            return false;
        }

        auto loan_result = publisher_->loan_uninit();
        if (!loan_result.has_value()) {
            spdlog::error("IoxPublisher failed to loan shared memory for service {}, error={}", service_name_, static_cast<int>(loan_result.error()));
            return false;
        }
        auto sample_uninit = std::move(loan_result).value();
        MessageType& payload = sample_uninit.payload_mut();  // 共享内存中的可写引用
        const bool ok = builder(payload);
        if (!ok) {
            spdlog::warn("IoxPublisher build callback failed for service {}", service_name_);
            return false;
        }

        auto sample_init = iox2::assume_init(std::move(sample_uninit));
        auto send_result = iox2::send(std::move(sample_init));
        if (!send_result.has_value()) {
            spdlog::error("IoxPublisher failed to send sample for service {}, error={}", service_name_, static_cast<int>(send_result.error()));
            return false;
        }

        ++published_count_;
        return true;
    }

    /**
     * @brief 设置/替换构建回调，允许运行时绑定数据填充逻辑
     * @param build_callback 共享内存构建回调
     * @return 设置成功返回 true
     */
    bool SetBuildCallback(BuildCallback build_callback)
    {
        if (!build_callback) {
            spdlog::error("IoxPublisher received null build callback for service {}", service_name_);
            return false;
        }
        build_callback_ = std::move(build_callback);
        return true;
    }

    /**
     * @brief 获取累计发布数量
     * @param 无
     * @return 已发布计数
     */
    uint64_t GetPublishedCount() const { return published_count_.load(); }

    /**
     * @brief 检查发布器是否就绪
     * @param 无
     * @return 就绪返回 true
     */
    bool IsReady() const { return publisher_.has_value(); }

  private:
    std::shared_ptr<IoxNode> node_;
    std::string service_name_;
    BuildCallback build_callback_;
    IoxPubSubConfig config_;
    std::atomic<uint64_t> published_count_;

    iox2::bb::Optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox2::bb::Optional<iox2::Publisher<iox2::ServiceType::Ipc, MessageType, void>> publisher_;
};

/**
 * @brief 基于 iceoryx2 的定长订阅器，支持同步拉取
 * @tparam MessageType 传输数据类型，需为平凡可拷贝
 */
template <typename MessageType>
class IoxSubscriber
{
  public:
    /// 零拷贝接收回调签名，直接访问共享内存
    using ReceiveCallback = std::function<void(const MessageType&)>;

    /**
     * @brief 构造订阅器
     * @param node 共享的 iceoryx2 节点
     * @param service_name 服务名称
     * @param receive_callback 接收回调，可为空
     * @param config 发布订阅配置
     * @return 无
     */
    IoxSubscriber(std::shared_ptr<IoxNode> node, std::string service_name, ReceiveCallback receive_callback = nullptr, IoxPubSubConfig config = {})
    : node_(std::move(node)),
      service_name_(std::move(service_name)),
      receive_callback_(std::move(receive_callback)),
      config_(config),
      received_count_(0)
    {
        static_assert(std::is_trivially_copyable_v<MessageType>, "MessageType 必须可平凡拷贝以支持零拷贝读取");

        if (!node_) {
            spdlog::error("IoxSubscriber requires a valid node, service={}", service_name_);
            return;
        }

        auto service_name_result = iox2::ServiceName::create(service_name_.c_str());
        if (!service_name_result.has_value()) {
            spdlog::error("Invalid iceoryx2 service name '{}', error={}", service_name_, static_cast<int>(service_name_result.error()));
            return;
        }
        auto iox_service_name = std::move(service_name_result).value();

        auto service_result = node_->service_builder(iox_service_name)
                                  .publish_subscribe<MessageType>()
                                  .max_publishers(config_.max_publishers)
                                  .max_subscribers(config_.max_subscribers)
                                  .subscriber_max_buffer_size(config_.subscriber_max_buffer_size)
                                  .history_size(config_.history_size)
                                  .enable_safe_overflow(config_.enable_safe_overflow)
                                  .open_or_create();

        if (!service_result.has_value()) {
            spdlog::error("Failed to open_or_create iceoryx2 service '{}', error={}", service_name_, static_cast<int>(service_result.error()));
            return;
        }
        service_.emplace(std::move(service_result).value());

        auto subscriber_result = service_->subscriber_builder().create();
        if (!subscriber_result.has_value()) {
            spdlog::error(
                "Failed to create iceoryx2 subscriber for service '{}', error={}",
                service_name_,
                static_cast<int>(subscriber_result.error()));
            return;
        }
        subscriber_.emplace(std::move(subscriber_result).value());

        spdlog::info(
            "IoxSubscriber ready: service={}, max_pub={}, max_sub={}, buffer={}",
            service_name_,
            config_.max_publishers,
            config_.max_subscribers,
            config_.subscriber_max_buffer_size);
    }

    /**
     * @brief 零拷贝拉取一条消息并在共享内存视图上处理
     * @param 无
     * @return 存在消息并处理成功返回 true，否则返回 false
     */
    bool Receive()
    {
        if (!subscriber_.has_value()) {
            spdlog::error("Subscriber not initialized for service {}", service_name_);
            return false;
        }
        if (!receive_callback_) {
            spdlog::error("Receive callback is null for service {}", service_name_);
            return false;
        }

        auto receive_result = subscriber_->receive();
        if (!receive_result.has_value()) {
            spdlog::error("Receive invocation failed for service {}, error={}", service_name_, static_cast<int>(receive_result.error()));
            return false;
        }
        auto sample = std::move(receive_result).value();
        if (!sample.has_value()) {
            return false;
        }

        const MessageType& payload = sample.value().payload();  // 直接引用共享内存中的数据
        receive_callback_(payload);
        ++received_count_;
        return true;
    }

    /**
     * @brief 零拷贝拉取当前所有可用消息并在共享内存视图上处理
     * @param 无
     * @return 已处理消息数量
     */
    uint64_t ReceiveAll()
    {
        if (!subscriber_.has_value()) {
            spdlog::error("Subscriber not initialized for service {}", service_name_);
            return 0;
        }
        if (!receive_callback_) {
            spdlog::error("Receive callback is null for service {}", service_name_);
            return 0;
        }

        uint64_t handled = 0;
        auto receive_result = subscriber_->receive();
        if (!receive_result.has_value()) {
            spdlog::error("Receive invocation failed for service {}, error={}", service_name_, static_cast<int>(receive_result.error()));
            return 0;
        }
        auto sample = std::move(receive_result).value();
        while (sample.has_value()) {
            const MessageType& payload = sample.value().payload();  // 直接引用共享内存中的数据
            receive_callback_(payload);
            ++received_count_;
            ++handled;
            receive_result = subscriber_->receive();
            if (!receive_result.has_value()) {
                spdlog::error("Receive invocation failed for service {}, error={}", service_name_, static_cast<int>(receive_result.error()));
                break;
            }
            sample = std::move(receive_result).value();
        }
        return handled;
    }

    /**
     * @brief 设置零拷贝接收回调
     * @param callback 零拷贝处理回调
     * @return 设置成功返回 true
     */
    bool SetReceiveCallback(ReceiveCallback callback)
    {
        if (!callback) {
            spdlog::error("Receive callback is null for service {}", service_name_);
            return false;
        }
        receive_callback_ = std::move(callback);
        return true;
    }

    /**
     * @brief 获取累计接收数量
     * @param 无
     * @return 已接收计数
     */
    uint64_t GetReceivedCount() const { return received_count_.load(); }

    /**
     * @brief 检查订阅器是否就绪
     * @param 无
     * @return 就绪返回 true
     */
    bool IsReady() const { return subscriber_.has_value(); }

  private:
    std::shared_ptr<IoxNode> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    IoxPubSubConfig config_;
    std::atomic<uint64_t> received_count_;

    iox2::bb::Optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox2::bb::Optional<iox2::Subscriber<iox2::ServiceType::Ipc, MessageType, void>> subscriber_;
};

}  // namespace ms_slam::slam_common
