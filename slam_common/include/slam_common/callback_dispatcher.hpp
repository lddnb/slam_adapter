#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
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
 * @brief 回调调度器，统一管理基于 iceoryx2 pub-sub 的轮询回调
 *
 * 通过注册 Poller 或订阅器对象，后台线程按优先级轮询，记录耗时统计并输出日志。
 */
class CallbackDispatcher
{
  public:
    /**
     * @brief 构造函数，初始化内部状态
     */
    CallbackDispatcher();

    /**
     * @brief 析构函数，确保工作线程安全退出
     */
    ~CallbackDispatcher();

    /**
     * @brief 回调统计信息
     */
    struct Stats
    {
        std::string name;                        ///< 回调名称
        int priority{0};                         ///< 调度优先级（大值先调度）
        uint64_t total_calls{0};                 ///< 总调用次数
        std::chrono::microseconds avg_duration;  ///< 平均耗时
    };

    /**
     * @brief 注册通用轮询函数
     * @param poll_func 非阻塞轮询函数，返回 true 表示处理过消息
     * @param name 调试名称
     * @param priority 优先级（越大越先执行）
     * @return 注册标识
     */
    uint64_t RegisterPoller(std::function<bool()> poll_func, const std::string& name = "poller", int priority = 0);

    /**
     * @brief 注册订阅器对象
     * @tparam SubscriberType 订阅器类型
     * @param subscriber 订阅器实例
     * @param name 调试名称
     * @param priority 优先级
     * @return 注册标识
     */
    template <typename SubscriberType>
    uint64_t RegisterSubscriber(std::shared_ptr<SubscriberType> subscriber,
                                const std::string& name = "subscriber",
                                int priority = 0)
    {
        auto poller = [subscriber]() -> bool {
            return subscriber->Receive();
        };
        return RegisterPoller(poller, name, priority);
    }

    /**
     * @brief 注册 RPC 服务对象，要求其提供 ReceiveAndRespond()
     * @tparam ServerType 服务类型
     * @param server 服务实例
     * @param name 调试名称
     * @param priority 优先级
     * @return 注册标识
     */
    template <typename ServerType>
    uint64_t RegisterServer(std::shared_ptr<ServerType> server, const std::string& name = "rpc_server", int priority = 0)
    {
        auto poller = [server]() -> bool { return server->ReceiveAndRespond(); };
        return RegisterPoller(poller, name, priority);
    }

    /**
     * @brief 注销指定注册项
     * @param id 注册标识
     * @return 成功返回 true
     */
    bool Unregister(uint64_t id);

    /**
     * @brief 启动后台轮询线程
     */
    void Start();

    /**
     * @brief 停止后台轮询线程
     */
    void Stop();

    /**
     * @brief 手动执行一次轮询
     */
    void PollOnce();

    /**
     * @brief 设置轮询间隔
     * @param interval 轮询间隔
     */
    void SetPollInterval(std::chrono::milliseconds interval);

    /**
     * @brief 获取所有注册项统计信息
     * @return 统计列表
     */
    std::vector<Stats> GetStatistics() const;

    /**
     * @brief 打印统计信息
     */
    void PrintStatistics() const;

    /**
     * @brief 清空全部注册项
     */
    void Clear();

    /**
     * @brief 当前注册数量
     * @return 注册数量
     */
    size_t Size() const;

  private:
    /**
     * @brief 内部回调条目
     */
    struct CallbackEntry
    {
        uint64_t id{0};                          ///< 注册标识
        std::function<bool()> poll_func;         ///< 轮询函数
        int priority{0};                         ///< 优先级
        std::string name;                        ///< 调试名称
        uint64_t total_calls{0};                 ///< 总调用次数
        std::chrono::nanoseconds total_duration{0};  ///< 累计耗时
    };

    void RunLoop();  ///< 后台线程主体

  private:
    std::vector<CallbackEntry> entries_;
    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    std::chrono::milliseconds poll_interval_{1};
    uint64_t next_id_{1};
};

}  // namespace ms_slam::slam_common
