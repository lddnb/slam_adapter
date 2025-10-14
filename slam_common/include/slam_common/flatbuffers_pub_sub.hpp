#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include <iox2/iceoryx2.hpp>
#include <flatbuffers/flatbuffers.h>
#include <spdlog/spdlog.h>

namespace ms_slam::slam_common
{

// ============================================================================
// 通用 FlatBuffers Publisher（非线程化，事件驱动）
// ============================================================================
struct PubSubConfig {
    uint32_t max_publishers{1};
    uint32_t max_subscribers{3};
    uint64_t initial_max_slice_len{16 * 1024 * 1024};  // 16MB default
    iox2::AllocationStrategy allocation_strategy{iox2::AllocationStrategy::Static};
    uint32_t subscriber_max_buffer_size{10};
    std::chrono::milliseconds poll_interval{3};
};

template <typename MessageType>
class FBSPublisher
{
  public:
    using PublishCallback = std::function<void(uint32_t seq, const MessageType&)>;

    FBSPublisher(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        PubSubConfig config = PubSubConfig())
    : node_(std::move(node)),
      service_name_(service_name),
      config_(config),
      published_count_(0)
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

    void set_publish_callback(PublishCallback callback)
    {
        publish_callback_ = callback;
    }

    bool publish(const MessageType& message)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffers Publisher service not initialized!");
        }

        try {
            // 使用消息类型自己的序列化方法
            auto buffer = message.serialize();

            // 分配动态大小的共享内存
            auto sample = publisher_->loan_slice_uninit(buffer.size()).expect("acquire sample");

            // 使用 write_from_fn 初始化数据
            auto initialized_sample = sample.write_from_fn([&buffer](uint64_t idx) {
                return buffer.data()[idx];
            });

            // 发送
            iox2::send(std::move(initialized_sample)).expect("send successful");

            published_count_++;

            // 触发回调
            if (publish_callback_) {
                publish_callback_(published_count_.load(), message);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffers publish error: " << e.what() << std::endl;
            return false;
        }
    }

    /// 直接发布已序列化的字节流 (零拷贝，用于Foxglove FlatBuffers消息)
    /// @param data 指向已序列化FlatBuffers数据的指针
    /// @param size 数据大小（字节）
    /// @return 发布成功返回true，否则返回false
    bool publish_raw(const uint8_t* data, size_t size)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffers Publisher service not initialized!");
        }

        try {
            // 分配动态大小的共享内存
            auto sample = publisher_->loan_slice_uninit(size).expect("acquire sample");

            // 使用 write_from_fn 初始化数据
            auto initialized_sample = sample.write_from_fn([data](uint64_t idx) {
                return data[idx];
            });

            // 发送
            iox2::send(std::move(initialized_sample)).expect("send successful");

            published_count_++;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffers publish_raw error: " << e.what() << std::endl;
            return false;
        }
    }

    /// 从 FlatBufferBuilder 直接发布（零拷贝）
    /// @param fbb FlatBufferBuilder对象
    /// @return 发布成功返回true，否则返回false
    bool publish_from_builder(flatbuffers::FlatBufferBuilder& fbb)
    {
        return publish_raw(fbb.GetBufferPointer(), fbb.GetSize());
    }

    uint32_t get_published_count() const { return published_count_.load(); }
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

// ============================================================================
// 通用 FlatBuffers Subscriber（非线程化，按需接收）
// ============================================================================

template <typename MessageType>
class FBSSubscriber
{
  public:
    using ReceiveCallback = std::function<void(const MessageType&)>;

    FBSSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback = nullptr,
        PubSubConfig config = PubSubConfig())
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      config_(config),
      received_count_(0)
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

    void set_receive_callback(ReceiveCallback callback)
    {
        receive_callback_ = callback;
    }

    iox::optional<MessageType> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                // 获取 payload (iox::Slice<uint8_t>)
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                // 使用消息类型自己的反序列化方法
                auto message = MessageType::deserialize(data, size);

                // 触发回调
                if (receive_callback_) {
                    receive_callback_(message);
                }

                return message;
            }
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffers receive error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

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

    /// 接收原始字节（单次，带拷贝）
    /// @return 返回原始字节的拷贝（避免悬空指针）
    /// @note 为了内存安全，必须拷贝数据。虽然有拷贝开销，但避免了FlatBuffers编解码
    iox::optional<std::vector<uint8_t>> receive_raw_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                // 获取 payload 并拷贝
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                return std::vector<uint8_t>(data, data + size);
            }
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffers receive_raw error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

    /// 接收所有原始字节（批量，带拷贝）
    /// @return 返回所有可用消息的原始字节拷贝
    /// @note 由于需要保存多个消息，必须拷贝数据以避免悬空指针
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

                // 获取 payload 并拷贝
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                raw_messages.emplace_back(data, data + size);

            } catch (const std::exception& e) {
                std::cerr << "FlatBuffers receive_all_raw error: " << e.what() << std::endl;
                break;
            }
        }

        return raw_messages;
    }

    bool has_data() const
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            return sample.has_value();
        } catch (const std::exception&) {
            return false;
        }
    }

    uint32_t get_received_count() const { return received_count_.load(); }
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

// ============================================================================
// 线程化 FlatBuffers Subscriber（后台线程持续轮询）
// ============================================================================

template <typename MessageType>
class ThreadedFBSSubscriber
{
  public:
    using ReceiveCallback = std::function<void(const MessageType&)>;

    ThreadedFBSSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback,
        PubSubConfig config = PubSubConfig())
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      config_(config),
      received_count_(0),
      running_(false),
      should_stop_(false)
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

    ~ThreadedFBSSubscriber()
    {
        stop();
    }

    void start()
    {
        if (!running_.load()) {
            should_stop_.store(false);
            thread_ = std::thread(&ThreadedFBSSubscriber::run, this);
            running_.store(true);
        }
    }

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

    bool is_running() const { return running_.load(); }

    void set_receive_callback(ReceiveCallback callback)
    {
        receive_callback_ = callback;
    }

    uint32_t get_received_count() const { return received_count_.load(); }
    bool is_ready() const { return subscriber_.has_value(); }

  private:
    void run()
    {
        while (!should_stop_.load()) {
            try {
                receive_messages();
                std::this_thread::sleep_for(config_.poll_interval);
            } catch (const std::exception& e) {
                std::cerr << "ThreadedFlatBuffer Subscriber error: " << e.what() << std::endl;
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

                // 获取 payload
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                // 反序列化
                auto message = MessageType::deserialize(data, size);

                // 触发回调（在独立线程中）
                if (receive_callback_) {
                    receive_callback_(message);
                }
            } catch (const std::exception& e) {
                std::cerr << "Message processing error: " << e.what() << std::endl;
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