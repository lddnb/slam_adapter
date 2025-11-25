/**
 * @file callback_dispatcher.hpp
 * @brief 回调调度器接口定义，统一管理各种轮询回调
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <ecal/msg/protobuf/subscriber.h>
#include <spdlog/spdlog.h>

namespace ms_slam::slam_common
{
/**
 * @brief 提供统一的回调调度能力
 */
class CallbackDispatcher
{
  public:
    /**
     * @brief 构造函数，初始化内部状态
     */
    CallbackDispatcher();

    /**
     * @brief 析构函数，自动停止工作线程
     */
    ~CallbackDispatcher();

    /**
     * @brief 注册通用轮询函数
     * @param poll_func 非阻塞轮询函数，返回是否处理消息
     * @param name 调试名称
     * @param priority 调度优先级
     * @return 注册标识符
     */
    uint64_t register_poller(std::function<bool()> poll_func, const std::string& name = "unnamed", int priority = 0);

    /**
     * @brief 注册 eCAL Protobuf 订阅者并记录回调耗时统计
     * @tparam Msg 消息类型
     * @param subscriber eCAL 订阅者
     * @param callback 业务回调，参数为 topic 名称、消息体、发送时间与时钟
     * @param name 调试名称
     * @param priority 调度优先级（仅用于统计展示）
     * @return 注册标识符
     */
    template <typename Msg>
    uint64_t RegisterEcalSubscriber(
        const std::shared_ptr<eCAL::protobuf::CSubscriber<Msg>>& subscriber,
        std::function<void(const std::string&, const Msg&, long long, long long)> callback,
        const std::string& name = "ecal_subscriber",
        int priority = 0)
    {
        struct QueuedMessage {
            std::string topic;  ///< 消息所属话题
            Msg message;        ///< 消息副本
            long long send_time;
            long long send_clock;
        };
        auto queue = std::make_shared<std::deque<QueuedMessage>>();
        auto queue_mutex = std::make_shared<std::mutex>();

        // 在 dispatcher 线程中消费队列，实现单线程处理
        auto poller = [queue, queue_mutex, callback]() -> bool {
            QueuedMessage queued{};
            {
                std::lock_guard<std::mutex> lock(*queue_mutex);
                if (queue->empty()) {
                    return false;
                }
                queued = std::move(queue->front());
                queue->pop_front();
            }
            try {
                if (callback) {
                    callback(queued.topic, queued.message, queued.send_time, queued.send_clock);
                }
            } catch (const std::exception& e) {
                spdlog::error("CallbackDispatcher queued callback threw exception: {}", e.what());
            }
            return true;
        };

        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t id = next_id_++;
        entries_.push_back({id, std::move(poller), priority, name, 0, std::chrono::nanoseconds(0)});
        std::sort(entries_.begin(), entries_.end(), [](const CallbackEntry& a, const CallbackEntry& b) { return a.priority > b.priority; });

        subscriber->SetReceiveCallback(
            [queue, queue_mutex, name](const eCAL::STopicId& topic_id, const Msg& msg, long long send_time, long long send_clock) {
                std::lock_guard<std::mutex> lock(*queue_mutex);
                try {
                    queue->push_back(QueuedMessage{topic_id.topic_name, msg, send_time, send_clock});
                } catch (const std::exception& e) {
                    spdlog::error("CallbackDispatcher subscriber '{}' enqueue failed: {}", name, e.what());
                }
            });

        spdlog::debug("Registered eCAL subscriber '{}' with id {}", name, id);
        return id;
    }

    /**
     * @brief 注销指定回调
     * @param id 注册标识符
     * @return 是否成功移除
     */
    bool unregister(uint64_t id);

    /**
     * @brief 启动后台调度线程
     */
    void start();

    /**
     * @brief 停止后台调度线程
     */
    void stop();

    /**
     * @brief 手动执行一次轮询
     */
    void poll_once();

    /**
     * @brief 设置轮询间隔
     * @param interval 轮询间隔
     */
    void set_poll_interval(std::chrono::milliseconds interval);

    /**
     * @brief 回调统计信息结构
     */
    struct Stats {
        std::string name;                        ///< 回调名称
        int priority;                            ///< 调度优先级
        uint64_t total_calls;                    ///< 总调用次数
        std::chrono::microseconds avg_duration;  ///< 平均处理耗时
    };

    /**
     * @brief 获取全部回调统计信息
     * @return 统计信息列表
     */
    std::vector<Stats> get_statistics() const;

    /**
     * @brief 打印统计信息
     */
    void print_statistics() const;

    /**
     * @brief 清空所有注册项
     */
    void clear();

    /**
     * @brief 获取当前注册数量
     * @return 注册数量
     */
    size_t size() const;

  private:
    /**
     * @brief 内部回调条目
     */
    struct CallbackEntry {
        uint64_t id;
        std::function<bool()> poll_func;
        int priority;
        std::string name;
        uint64_t total_calls;
        std::chrono::nanoseconds total_duration{0};
    };

    std::vector<CallbackEntry> entries_;
    mutable std::mutex mutex_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    std::chrono::milliseconds poll_interval_;
    uint64_t next_id_;
};

}  // namespace ms_slam::slam_common
