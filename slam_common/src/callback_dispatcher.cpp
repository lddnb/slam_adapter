/**
 * @file callback_dispatcher.cpp
 * @brief 回调调度器实现
 */

#include "slam_common/callback_dispatcher.hpp"

#include <algorithm>
#include <utility>

#include <spdlog/stopwatch.h>

namespace ms_slam::slam_common
{

CallbackDispatcher::CallbackDispatcher()
: running_(false)
, poll_interval_(std::chrono::milliseconds(1))
, next_id_(1)
{
}

CallbackDispatcher::~CallbackDispatcher()
{
    stop();
}

uint64_t CallbackDispatcher::register_poller(std::function<bool()> poll_func,
                                             const std::string& name,
                                             int priority)
{
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t id = next_id_++;
    entries_.push_back({id, std::move(poll_func), priority, name, 0});

    std::sort(entries_.begin(), entries_.end(), [](const CallbackEntry& a, const CallbackEntry& b) {
        return a.priority > b.priority;
    });

    spdlog::debug("Registered poller '{}' with id {}", name, id);
    return id;
}

bool CallbackDispatcher::unregister(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(entries_.begin(), entries_.end(), [id](const CallbackEntry& entry) {
        return entry.id == id;
    });

    if (it != entries_.end()) {
        spdlog::debug("Unregistered poller '{}' with id {}", it->name, id);
        entries_.erase(it);
        return true;
    }

    return false;
}

void CallbackDispatcher::start()
{
    if (running_.exchange(true)) {
        return;
    }

    spdlog::info("Starting Callback Dispatcher...");
    worker_thread_ = std::thread([this]() {
        while (running_) {
            poll_once();
            std::this_thread::sleep_for(poll_interval_);
        }
    });
}

void CallbackDispatcher::stop()
{
    if (!running_.exchange(false)) {
        return;
    }

    spdlog::info("Stopping Callback Dispatcher...");

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void CallbackDispatcher::poll_once()
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& entry : entries_) {
        spdlog::stopwatch sw;
        bool processed = entry.poll_func();
        auto duration = duration_cast<std::chrono::nanoseconds>(sw.elapsed());

        if (processed) {
            entry.total_calls++;
            entry.total_duration += duration;
        }
    }
}

void CallbackDispatcher::set_poll_interval(std::chrono::milliseconds interval)
{
    poll_interval_ = interval;
    spdlog::debug("Set poll interval to {} ms", interval.count());
}

std::vector<CallbackDispatcher::Stats> CallbackDispatcher::get_statistics() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Stats> stats;
    stats.reserve(entries_.size());

    for (const auto& entry : entries_) {
        Stats stat{};
        stat.name = entry.name;
        stat.priority = entry.priority;
        stat.total_calls = entry.total_calls;

        if (entry.total_calls > 0) {
            stat.avg_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                entry.total_duration / entry.total_calls);
        } else {
            stat.avg_duration = std::chrono::microseconds(0);
        }

        stats.push_back(stat);
    }

    return stats;
}

void CallbackDispatcher::print_statistics() const
{
    auto stats = get_statistics();

    spdlog::info("=== CallbackDispatcher Statistics ===");
    spdlog::info("Name                 | Priority | Calls    | Avg Duration");
    spdlog::info("---------------------+----------+----------+-------------");

    for (const auto& s : stats) {
        spdlog::info("{:<20} | {:>8} | {:>8} | {:>8} us",
                      s.name,
                      s.priority,
                      s.total_calls,
                      s.avg_duration.count());
    }

    spdlog::info("");
}

void CallbackDispatcher::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    spdlog::debug("Cleared all registered pollers");
}

size_t CallbackDispatcher::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

}  // namespace ms_slam::slam_common

