#pragma once

#include "slam_common/slam_concepts.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <iostream>
#include <concepts>

#include <iox2/iceoryx2.hpp>

namespace ms_slam::slam_common
{
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
        MessageGenerator message_generator,
        PublisherConfig config = {})
    : node_(std::move(node)),
      service_name_(service_name),
      message_generator_(std::move(message_generator)),
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

// Convenience typedefs for common message types with C++20 concepts
#include "iceoryx_data_types.hpp"

// Template aliases for different point types with concept constraints
template <BasicPoint PointType>
using PointCloudPublisher = ThreadedPublisher<PointCloud<PointType>>;

template <BasicPoint PointType>
using PointCloudSubscriber = ThreadedSubscriber<PointCloud<PointType>>;

using ImagePublisher = ThreadedPublisher<Image>;
using ImageSubscriber = ThreadedSubscriber<Image>;

// Specific point type aliases for convenience
using PointITCloudPublisher = PointCloudPublisher<PointIT>;
using PointITCloudSubscriber = PointCloudSubscriber<PointIT>;
using PointRGBCloudPublisher = PointCloudPublisher<PointRGB>;
using PointRGBCloudSubscriber = PointCloudSubscriber<PointRGB>;
using PointRGBITCloudPublisher = PointCloudPublisher<PointRGBIT>;
using PointRGBITCloudSubscriber = PointCloudSubscriber<PointRGBIT>;

}  // namespace ms_slam::slam_common