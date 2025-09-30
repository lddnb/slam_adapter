#pragma once

#include "slam_common/slam_concepts.hpp"
#include "slam_common/iceoryx_data_types.hpp"
#include "slam_common/flatbuffer_serializer.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <iostream>
#include <concepts>
#include <vector>
#include <cstring>

#include <iox2/iceoryx2.hpp>
#include <flatbuffers/flatbuffers.h>

namespace ms_slam::slam_common
{
// Forward declarations
template <Message MessageType> class DirectPublisher;
template <Message MessageType> class DirectSubscriber;
struct DynamicImage; // Forward declaration for dynamic image classes

// Base class for threaded components
class ThreadedComponent
{
  public:
    ThreadedComponent() : running_(false), should_stop_(false) {}

    virtual ~ThreadedComponent() { stop(); }

    void start()
    {
        if (!running_.load()) {
            should_stop_.store(false);
            thread_ = std::thread(&ThreadedComponent::run, this);
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

  protected:
    virtual void run() = 0;

    bool should_stop() const { return should_stop_.load(); }

  private:
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;
    std::thread thread_;
};

// C++20 Enhanced Generic Threaded Publisher with concepts
template <Message MessageType>
class ThreadedPublisher : public ThreadedComponent
{
  public:
    using PublishCallback = std::function<void(uint32_t seq, const MessageType&)>;
    using MessageGenerator = std::function<MessageType(uint32_t seq)>;

    // C++20 designated initializers would be nice here, but let's use structured binding
    struct PublisherConfig {
        std::chrono::milliseconds publish_interval{100};
        uint32_t max_publishers{1};
        uint32_t max_subscribers{10};
    };

    ThreadedPublisher(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        PublisherConfig config = {})
    : node_(std::move(node)),
      service_name_(service_name),
      config_(config),
      seq_counter_(0)
    {
        setup_service();
    }

    // C++20 feature: explicit overloads with concepts
    template <std::invocable<uint32_t> Generator>
        requires std::same_as<std::invoke_result_t<Generator, uint32_t>, MessageType>
    void set_message_generator(Generator&& generator)
    {
        std::lock_guard<std::mutex> lock(generator_mutex_);
        message_generator_ = std::forward<Generator>(generator);
    }

    void set_publish_callback(PublishCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        publish_callback_ = callback;
    }

    void set_message_generator(MessageGenerator generator)
    {
        std::lock_guard<std::mutex> lock(generator_mutex_);
        message_generator_ = generator;
    }

    uint32_t get_published_count() const { return seq_counter_.load(); }

    // Publish a single message immediately (non-threaded mode)
    bool publish(const MessageType& message)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("Publisher service not initialized!");
        }

        auto sample = publisher_->loan_uninit().expect("acquire sample");
        auto initialized_sample = sample.write_payload(MessageType(message));
        iox2::send(std::move(initialized_sample)).expect("send successful");

        uint32_t seq = ++seq_counter_;

        // Notify callback if set
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (publish_callback_) {
            publish_callback_(seq, message);
        }

        return true;
    }

  protected:
    void run() override
    {
        while (!should_stop()) {
            try {
                publish_generated_message();
                std::this_thread::sleep_for(config_.publish_interval);
            } catch (const std::exception& e) {
                std::cerr << "Publisher error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

  private:
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<MessageType>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .open_or_create()
                       .expect("successful service creation");

        publisher_ = service_->publisher_builder().create().expect("successful publisher creation");
    }

    void publish_generated_message()
    {
        std::lock_guard<std::mutex> lock(generator_mutex_);
        if (!message_generator_) {
            return;
        }

        uint32_t seq = ++seq_counter_;
        auto message = message_generator_(seq);

        {
            std::lock_guard<std::mutex> callback_lock(callback_mutex_);
            if (publish_callback_) {
                publish_callback_(seq, message);
            }
        }

        auto sample = publisher_->loan_uninit().expect("acquire sample");
        auto initialized_sample = sample.write_payload(std::move(message));
        iox2::send(std::move(initialized_sample)).expect("send successful");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    MessageGenerator message_generator_;
    PublisherConfig config_;
    std::atomic<uint32_t> seq_counter_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox::optional<iox2::Publisher<iox2::ServiceType::Ipc, MessageType, void>> publisher_;

    std::mutex callback_mutex_;
    std::mutex generator_mutex_;
    PublishCallback publish_callback_;
};

// Generic Threaded Subscriber - similar to ros::Subscriber
template <typename MessageType>
class ThreadedSubscriber : public ThreadedComponent
{
  public:
    using ReceiveCallback = std::function<void(const MessageType&)>;

    ThreadedSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback,
        std::chrono::milliseconds poll_interval = std::chrono::milliseconds(10))
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      poll_interval_(poll_interval),
      received_count_(0)
    {
        setup_service();
    }

    void set_receive_callback(ReceiveCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        receive_callback_ = callback;
    }

    uint32_t get_received_count() const { return received_count_.load(); }

    // Receive a single message immediately (non-threaded mode)
    iox::optional<MessageType> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;
                return MessageType(sample->payload());
            }
        } catch (const std::exception& e) {
            std::cerr << "Receive error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

  protected:
    void run() override
    {
        while (!should_stop()) {
            try {
                receive_messages();
                std::this_thread::sleep_for(poll_interval_);
            } catch (const std::exception& e) {
                std::cerr << "Subscriber error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

  private:
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name).publish_subscribe<MessageType>().open_or_create().expect("successful service creation");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");
    }

    void receive_messages()
    {
        auto sample = subscriber_->receive().expect("receive succeeds");
        while (sample.has_value()) {
            received_count_++;
            const auto& message = sample->payload();

            // Notify callback if set
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (receive_callback_) {
                receive_callback_(message);
            }

            sample = subscriber_->receive().expect("receive succeeds");
        }
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    std::chrono::milliseconds poll_interval_;
    std::atomic<uint32_t> received_count_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, MessageType, void>> subscriber_;

    std::mutex callback_mutex_;
};

// Direct (non-threaded) Publisher for event-driven publishing
template <Message MessageType>
class DirectPublisher
{
  public:
    using PublishCallback = std::function<void(uint32_t seq, const MessageType&)>;

    struct PublisherConfig {
        uint32_t max_publishers{1};
        uint32_t max_subscribers{10};
    };

    DirectPublisher(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        PublisherConfig config = {})
    : node_(std::move(node)),
      service_name_(service_name),
      config_(config),
      seq_counter_(0)
    {
        setup_service();
    }

    void set_publish_callback(PublishCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        publish_callback_ = callback;
    }

    // Synchronous publish - optimized for iceoryx2's low-latency characteristics
    bool publish(const MessageType& message)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("Publisher service not initialized!");
        }

        try {
            auto sample = publisher_->loan_uninit().expect("acquire sample");
            auto initialized_sample = sample.write_payload(MessageType(message));
            iox2::send(std::move(initialized_sample)).expect("send successful");

            uint32_t seq = ++seq_counter_;

            // Notify callback if set
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (publish_callback_) {
                publish_callback_(seq, message);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Direct publish error: " << e.what() << std::endl;
            return false;
        }
    }

    // Batch publish for multiple messages
    template<typename Iterator>
    size_t publish_batch(Iterator begin, Iterator end)
    {
        size_t published_count = 0;
        for (auto it = begin; it != end; ++it) {
            if (publish(*it)) {
                ++published_count;
            }
        }
        return published_count;
    }

    uint32_t get_published_count() const { return seq_counter_.load(); }

    bool is_ready() const { return publisher_.has_value(); }

  private:
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<MessageType>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .open_or_create()
                       .expect("successful service creation");

        publisher_ = service_->publisher_builder().create().expect("successful publisher creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    PublisherConfig config_;
    std::atomic<uint32_t> seq_counter_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox::optional<iox2::Publisher<iox2::ServiceType::Ipc, MessageType, void>> publisher_;

    std::mutex callback_mutex_;
    PublishCallback publish_callback_;
};

// Direct (non-threaded) Subscriber for on-demand receiving
template <Message MessageType>
class DirectSubscriber
{
  public:
    using ReceiveCallback = std::function<void(const MessageType&)>;

    DirectSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback = nullptr)
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      received_count_(0)
    {
        setup_service();
    }

    void set_receive_callback(ReceiveCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        receive_callback_ = callback;
    }

    // Receive a single message (non-blocking)
    iox::optional<MessageType> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;
                auto message = MessageType(sample->payload());

                // Notify callback if set
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (receive_callback_) {
                    receive_callback_(message);
                }

                return message;
            }
        } catch (const std::exception& e) {
            std::cerr << "Direct receive error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

    // Receive all available messages
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

    // Check if messages are available without consuming them
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
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<MessageType>()
                       .open_or_create()
                       .expect("successful service creation");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    std::atomic<uint32_t> received_count_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, MessageType, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, MessageType, void>> subscriber_;

    std::mutex callback_mutex_;
};


// Template aliases for different point types with concept constraints

// Threaded (asynchronous) publishers and subscribers
// Note: These use PointCloudSoA instead of PointCloud concept
// template <BasicPoint PointType>
// using PointCloudPublisher = ThreadedPublisher<PointCloudSoA<PointType>>;

// template <BasicPoint PointType>
// using PointCloudSubscriber = ThreadedSubscriber<PointCloudSoA<PointType>>;

// Direct (synchronous) publishers and subscribers for event-driven scenarios
// template <BasicPoint PointType>
// using DirectPointCloudPublisher = DirectPublisher<PointCloudSoA<PointType>>;

// template <BasicPoint PointType>
// using DirectPointCloudSubscriber = DirectSubscriber<PointCloudSoA<PointType>>;

// Specific point type aliases for convenience
// Commented out - use FlatBuffer versions instead
// using PointITCloudPublisher = PointCloudPublisher<PointIT>;
// using PointITCloudSubscriber = PointCloudSubscriber<PointIT>;
// using PointRGBCloudPublisher = PointCloudPublisher<PointRGB>;
// using PointRGBCloudSubscriber = PointCloudSubscriber<PointRGB>;
// using PointRGBITCloudPublisher = PointCloudPublisher<PointRGBIT>;
// using PointRGBITCloudSubscriber = PointCloudSubscriber<PointRGBIT>;

// Direct versions - commented out
// using DirectPointITCloudPublisher = DirectPointCloudPublisher<PointIT>;
// using DirectPointITCloudSubscriber = DirectPointCloudSubscriber<PointIT>;
// using DirectPointRGBCloudPublisher = DirectPointCloudPublisher<PointRGB>;
// using DirectPointRGBCloudSubscriber = DirectPointCloudSubscriber<PointRGB>;
// using DirectPointRGBITCloudPublisher = DirectPointCloudPublisher<PointRGBIT>;
// using DirectPointRGBITCloudSubscriber = DirectPointCloudSubscriber<PointRGBIT>;

// ============ FlatBuffers-based Image Publisher/Subscriber ============
// These use dynamic sizing through FlatBuffers serialization with iox::Slice<uint8_t>

class FlatBufferImagePublisher
{
  public:
    using PublishCallback = std::function<void(uint32_t seq, const Image&)>;

    struct PublisherConfig {
        uint32_t max_publishers{1};
        uint32_t max_subscribers{10};
        uint64_t initial_max_slice_len{16 * 1024 * 1024};  // 16MB for large images (2448x2048 RGB ~15MB)
        iox2::AllocationStrategy allocation_strategy{iox2::AllocationStrategy::Static};  // Static for predictable performance
    };

    FlatBufferImagePublisher(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        PublisherConfig config = {1, 10, 16 * 1024 * 1024, iox2::AllocationStrategy::Static})
    : node_(std::move(node)),
      service_name_(service_name),
      config_(config),
      seq_counter_(0)
    {
        setup_service();
    }

    void set_publish_callback(PublishCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        publish_callback_ = callback;
    }

    bool publish(const Image& image)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffer Image Publisher service not initialized!");
        }

        try {
            // 序列化为 FlatBuffers
            auto buffer = ImageSerializer::serialize(image);

            // 分配动态大小的共享内存 (使用 iox::Slice<uint8_t>)
            auto sample = publisher_->loan_slice_uninit(buffer.size()).expect("acquire sample");

            // 使用 write_from_fn 初始化数据
            auto initialized_sample = sample.write_from_fn([&buffer](uint64_t idx) {
                return buffer.data()[idx];
            });

            // 发送
            iox2::send(std::move(initialized_sample)).expect("send successful");

            uint32_t seq = ++seq_counter_;

            // Notify callback if set
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (publish_callback_) {
                publish_callback_(seq, image);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffer Image publish error: " << e.what() << std::endl;
            return false;
        }
    }

    uint32_t get_published_count() const { return seq_counter_.load(); }
    bool is_ready() const { return publisher_.has_value(); }

  private:
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .open_or_create()
                       .expect("successful service creation");

        publisher_ = service_->publisher_builder()
                         .initial_max_slice_len(config_.initial_max_slice_len)
                         .allocation_strategy(config_.allocation_strategy)
                         .create()
                         .expect("successful publisher creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    PublisherConfig config_;
    std::atomic<uint32_t> seq_counter_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Publisher<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> publisher_;

    std::mutex callback_mutex_;
    PublishCallback publish_callback_;
};

class FlatBufferImageSubscriber
{
  public:
    using ReceiveCallback = std::function<void(const Image&)>;

    FlatBufferImageSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback = nullptr,
        std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      received_count_(0),
      memory_resource_(mr)
    {
        setup_service();
    }

    void set_receive_callback(ReceiveCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        receive_callback_ = callback;
    }

    iox::optional<Image> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                // 获取 payload (iox::Slice<uint8_t>)
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                // 零拷贝反序列化
                auto image = ImageSerializer::deserialize(data, size, memory_resource_);

                // Notify callback if set
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (receive_callback_) {
                    receive_callback_(image);
                }

                return image;
            }
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffer Image receive error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

    std::vector<Image> receive_all()
    {
        std::vector<Image> images;
        auto image = receive_once();
        while (image.has_value()) {
            images.push_back(std::move(image.value()));
            image = receive_once();
        }
        return images;
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
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .open_or_create()
                       .expect("successful service creation");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    std::atomic<uint32_t> received_count_;
    std::pmr::memory_resource* memory_resource_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> subscriber_;

    std::mutex callback_mutex_;
};

// ============ FlatBuffers-based PointCloud Publisher/Subscriber (模板) ============

template <typename PointType>
class FlatBufferPointCloudPublisher
{
  public:
    using PointCloudType = PointCloudSoA<PointType>;
    using PublishCallback = std::function<void(uint32_t seq, const PointCloudType&)>;

    struct PublisherConfig {
        uint32_t max_publishers{1};
        uint32_t max_subscribers{10};
        uint64_t initial_max_slice_len{16 * 1024 * 1024};  // 16MB for large point clouds
        iox2::AllocationStrategy allocation_strategy{iox2::AllocationStrategy::Static};  // Static for predictable performance
    };

    FlatBufferPointCloudPublisher(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        PublisherConfig config = {1, 10, 16 * 1024 * 1024, iox2::AllocationStrategy::Static})
    : node_(std::move(node)),
      service_name_(service_name),
      config_(config),
      seq_counter_(0)
    {
        setup_service();
    }

    void set_publish_callback(PublishCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        publish_callback_ = callback;
    }

    bool publish(const PointCloudType& cloud)
    {
        if (!publisher_.has_value()) {
            throw std::runtime_error("FlatBuffer PointCloud Publisher service not initialized!");
        }

        try {
            // 序列化为 FlatBuffers
            auto buffer = PointCloudSerializer<PointType>::serialize(cloud);

            // 分配动态大小的共享内存
            auto sample = publisher_->loan_slice_uninit(buffer.size()).expect("acquire sample");

            // 使用 write_from_fn 初始化数据
            auto initialized_sample = sample.write_from_fn([&buffer](uint64_t idx) {
                return buffer.data()[idx];
            });

            // 发送
            iox2::send(std::move(initialized_sample)).expect("send successful");

            uint32_t seq = ++seq_counter_;

            // Notify callback if set
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (publish_callback_) {
                publish_callback_(seq, cloud);
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffer PointCloud publish error: " << e.what() << std::endl;
            return false;
        }
    }

    uint32_t get_published_count() const { return seq_counter_.load(); }
    bool is_ready() const { return publisher_.has_value(); }

  private:
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .max_publishers(config_.max_publishers)
                       .max_subscribers(config_.max_subscribers)
                       .open_or_create()
                       .expect("successful service creation");

        publisher_ = service_->publisher_builder()
                         .initial_max_slice_len(config_.initial_max_slice_len)
                         .allocation_strategy(config_.allocation_strategy)
                         .create()
                         .expect("successful publisher creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    PublisherConfig config_;
    std::atomic<uint32_t> seq_counter_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Publisher<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> publisher_;

    std::mutex callback_mutex_;
    PublishCallback publish_callback_;
};

template <typename PointType>
class FlatBufferPointCloudSubscriber
{
  public:
    using PointCloudType = PointCloudSoA<PointType>;
    using ReceiveCallback = std::function<void(const PointCloudType&)>;

    FlatBufferPointCloudSubscriber(
        std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node,
        const std::string& service_name,
        ReceiveCallback receive_callback = nullptr,
        std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    : node_(node),
      service_name_(service_name),
      receive_callback_(receive_callback),
      received_count_(0),
      memory_resource_(mr)
    {
        setup_service();
    }

    void set_receive_callback(ReceiveCallback callback)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        receive_callback_ = callback;
    }

    iox::optional<PointCloudType> receive_once()
    {
        try {
            auto sample = subscriber_->receive().expect("receive succeeds");
            if (sample.has_value()) {
                received_count_++;

                // 获取 payload (iox::Slice<uint8_t>)
                auto payload = sample->payload();
                const uint8_t* data = payload.begin();
                size_t size = payload.number_of_bytes();

                // 零拷贝反序列化
                auto cloud = PointCloudSerializer<PointType>::deserialize(data, size, memory_resource_);

                // Notify callback if set
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (receive_callback_) {
                    receive_callback_(cloud);
                }

                return cloud;
            }
        } catch (const std::exception& e) {
            std::cerr << "FlatBuffer PointCloud receive error: " << e.what() << std::endl;
        }
        return iox::nullopt;
    }

    std::vector<PointCloudType> receive_all()
    {
        std::vector<PointCloudType> clouds;
        auto cloud = receive_once();
        while (cloud.has_value()) {
            clouds.push_back(std::move(cloud.value()));
            cloud = receive_once();
        }
        return clouds;
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
    void setup_service()
    {
        auto service_name = iox2::ServiceName::create(service_name_.c_str()).expect("valid service name");
        service_ = node_->service_builder(service_name)
                       .publish_subscribe<iox::Slice<uint8_t>>()
                       .open_or_create()
                       .expect("successful service creation");

        subscriber_ = service_->subscriber_builder().create().expect("successful subscriber creation");
    }

    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;
    std::string service_name_;
    ReceiveCallback receive_callback_;
    std::atomic<uint32_t> received_count_;
    std::pmr::memory_resource* memory_resource_;

    iox::optional<iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> service_;
    iox::optional<iox2::Subscriber<iox2::ServiceType::Ipc, iox::Slice<uint8_t>, void>> subscriber_;

    std::mutex callback_mutex_;
};

// Convenience type aliases for FlatBuffers-based publishers/subscribers
using FlatBufferPointITPublisher = FlatBufferPointCloudPublisher<PointIT>;
using FlatBufferPointITSubscriber = FlatBufferPointCloudSubscriber<PointIT>;
using FlatBufferPointRGBPublisher = FlatBufferPointCloudPublisher<PointRGB>;
using FlatBufferPointRGBSubscriber = FlatBufferPointCloudSubscriber<PointRGB>;
using FlatBufferPointRGBITPublisher = FlatBufferPointCloudPublisher<PointRGBIT>;
using FlatBufferPointRGBITSubscriber = FlatBufferPointCloudSubscriber<PointRGBIT>;

}  // namespace ms_slam::slam_common