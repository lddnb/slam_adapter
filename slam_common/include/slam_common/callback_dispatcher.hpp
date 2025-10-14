#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>

#include <spdlog/spdlog.h>

namespace ms_slam::slam_common
{

// ============================================================================
// CallbackDispatcher: 统一管理和调度各种 subscriber 和 server 的回调
// ============================================================================

class CallbackDispatcher
{
  public:
    CallbackDispatcher()
    : running_(false)
    , poll_interval_(std::chrono::milliseconds(1))
    , next_id_(1)
    {
    }

    ~CallbackDispatcher()
    {
        stop();
    }

    // 注册一个通用的轮询函数
    // poll_func: 轮询函数，应该非阻塞地检查并处理一个消息，返回是否处理了消息
    // name: 用于调试和日志
    // priority: 优先级，数值越大优先级越高
    uint64_t register_poller(std::function<bool()> poll_func,
                            const std::string& name = "unnamed",
                            int priority = 0)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        uint64_t id = next_id_++;
        entries_.push_back({id, std::move(poll_func), priority, name, 0, 0});

        // 按优先级排序（高优先级在前）
        std::sort(entries_.begin(), entries_.end(),
                 [](const CallbackEntry& a, const CallbackEntry& b) {
                     return a.priority > b.priority;
                 });

        return id;
    }

    // 注册 Subscriber（带回调）
    template<typename T>
    uint64_t register_subscriber(std::shared_ptr<T> subscriber,
                                const std::string& name = "subscriber",
                                int priority = 0)
    {
        auto poll_func = [subscriber, name]() -> bool {
            try {
                auto sample = subscriber->receive_once();
                return sample.has_value();  // 返回是否接收到消息
            } catch (const std::exception& e) {
                spdlog::error("[CallbackDispatcher] Subscriber '{}' error: {}", name, e.what());
                return false;
            }
        };

        return register_poller(poll_func, name, priority);
    }

    // 注册 RPCServer（需要事先设置 callback）
    template<typename ServerType>
    uint64_t register_server(std::shared_ptr<ServerType> server,
                            const std::string& name = "rpc_server",
                            int priority = 0)
    {
        auto poll_func = [server, name]() -> bool {
            try {
                return server->receive_and_respond();  // 返回是否处理了请求
            } catch (const std::exception& e) {
                spdlog::error("[CallbackDispatcher] Server '{}' error: {}", name, e.what());
                return false;
            }
        };

        return register_poller(poll_func, name, priority);
    }

    // 移除注册的回调
    bool unregister(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(entries_.begin(), entries_.end(),
                              [id](const CallbackEntry& e) { return e.id == id; });

        if (it != entries_.end()) {
            entries_.erase(it);
            return true;
        }
        return false;
    }

    // 启动调度线程
    void start()
    {
        if (running_.exchange(true)) {
            return;  // 已经在运行
        }

        spdlog::info("Starting Callback Dispatcher...");
        worker_thread_ = std::thread([this]() {
            while (running_) {
                poll_once();
                std::this_thread::sleep_for(poll_interval_);
            }
        });
    }

    // 停止调度线程
    void stop()
    {
        if (!running_.exchange(false)) {
            return;  // 已经停止
        }

        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // 手动执行一次轮询（用于测试或手动模式）
    void poll_once()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& entry : entries_) {
            try {
                auto start = std::chrono::steady_clock::now();

                bool processed = entry.poll_func();

                auto duration = std::chrono::steady_clock::now() - start;

                // 只有实际处理了消息时才递增计数
                if (processed) {
                    entry.total_calls++;
                    entry.total_duration += duration;
                }

            } catch (const std::exception& e) {
                spdlog::error("[CallbackDispatcher] Uncaught exception in '{}': {}", entry.name, e.what());
                entry.total_errors++;
            }
        }
    }

    // 设置轮询间隔
    void set_poll_interval(std::chrono::milliseconds interval)
    {
        poll_interval_ = interval;
    }

    // 获取统计信息
    struct Stats {
        std::string name;
        int priority;
        uint64_t total_calls;
        uint64_t total_errors;
        std::chrono::microseconds avg_duration;
    };

    std::vector<Stats> get_statistics() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Stats> stats;
        for (const auto& entry : entries_) {
            Stats s;
            s.name = entry.name;
            s.priority = entry.priority;
            s.total_calls = entry.total_calls;
            s.total_errors = entry.total_errors;

            if (entry.total_calls > 0) {
                s.avg_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    entry.total_duration / entry.total_calls);
            } else {
                s.avg_duration = std::chrono::microseconds(0);
            }

            stats.push_back(s);
        }

        return stats;
    }

    // 打印统计信息
    void print_statistics() const
    {
        auto stats = get_statistics();

        spdlog::info("=== CallbackDispatcher Statistics ===");
        spdlog::info("Name                 | Priority | Calls    | Errors | Avg Duration");
        spdlog::info("---------------------+----------+----------+--------+-------------");

        for (const auto& s : stats) {
            spdlog::info("{:<20} | {:>8} | {:>8} | {:>6} | {:>8} us",
                   s.name, s.priority, s.total_calls, s.total_errors,
                   s.avg_duration.count());
        }
        spdlog::info("");
    }

    // 清空所有注册
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

    // 获取注册数量
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

  private:
    struct CallbackEntry {
        uint64_t id;
        std::function<bool()> poll_func;
        int priority;
        std::string name;
        uint64_t total_calls;
        uint64_t total_errors;
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
