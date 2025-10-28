/**
 * @file callback_dispatcher.hpp
 * @brief 回调调度器接口定义，统一管理各种轮询回调
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
     * @brief 注册 subscriber 对象
     * @tparam T 订阅器类型
     * @param subscriber 订阅器实例
     * @param name 调试名称
     * @param priority 调度优先级
     * @return 注册标识符
     */
    template <typename T>
    uint64_t register_subscriber(std::shared_ptr<T> subscriber, const std::string& name = "subscriber", int priority = 0)
    {
        auto poll_func = [subscriber, name]() -> bool {
            auto sample = subscriber->receive_once();
            return sample.has_value();
        };

        return register_poller(poll_func, name, priority);
    }

    /**
     * @brief 注册 RPC server 对象
     * @tparam ServerType 服务类型
     * @param server 服务实例
     * @param name 调试名称
     * @param priority 调度优先级
     * @return 注册标识符
     */
    template <typename ServerType>
    uint64_t register_server(std::shared_ptr<ServerType> server, const std::string& name = "rpc_server", int priority = 0)
    {
        auto poll_func = [server, name]() -> bool { return server->receive_and_respond(); };

        return register_poller(poll_func, name, priority);
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
