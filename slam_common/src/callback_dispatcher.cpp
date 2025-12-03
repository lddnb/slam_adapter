#include "slam_common/callback_dispatcher.hpp"

#include <algorithm>
#include <utility>

namespace ms_slam::slam_common
{

CallbackDispatcher::CallbackDispatcher() = default;

CallbackDispatcher::~CallbackDispatcher()
{
    Stop();
}

uint64_t CallbackDispatcher::RegisterPoller(std::function<bool()> poll_func, const std::string& name, int priority)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t id = next_id_++;
    CallbackEntry entry;
    entry.id = id;
    entry.poll_func = std::move(poll_func);
    entry.priority = priority;
    entry.name = name;

    entries_.push_back(std::move(entry));
    std::sort(entries_.begin(), entries_.end(), [](const CallbackEntry& lhs, const CallbackEntry& rhs) {
        return lhs.priority > rhs.priority;
    });

    spdlog::info("CallbackDispatcher registered poller '{}' with id={} priority={}", name, id, priority);
    return id;
}

bool CallbackDispatcher::Unregister(uint64_t id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto before = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(),
                                  entries_.end(),
                                  [id](const CallbackEntry& entry) { return entry.id == id; }),
                   entries_.end());
    const bool removed = entries_.size() != before;
    if (removed) {
        spdlog::info("CallbackDispatcher unregistered id={}", id);
    }
    return removed;
}

void CallbackDispatcher::Start()
{
    if (running_.load()) {
        return;
    }
    running_.store(true);
    worker_thread_ = std::thread(&CallbackDispatcher::RunLoop, this);
    spdlog::info("CallbackDispatcher started");
}

void CallbackDispatcher::Stop()
{
    if (!running_.load()) {
        return;
    }
    running_.store(false);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    spdlog::info("CallbackDispatcher stopped");
}

void CallbackDispatcher::PollOnce()
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& entry : entries_) {
        const auto start = std::chrono::steady_clock::now();
        bool handled = false;
        try {
            handled = entry.poll_func();
        } catch (const std::exception& e) {
            spdlog::error("CallbackDispatcher poller '{}' threw exception: {}", entry.name, e.what());
        } catch (...) {
            spdlog::error("CallbackDispatcher poller '{}' threw unknown exception", entry.name);
        }
        const auto duration = std::chrono::steady_clock::now() - start;

        if (handled) {
            entry.total_duration += duration;
            entry.total_calls += 1;
        }
    }
}

void CallbackDispatcher::SetPollInterval(std::chrono::milliseconds interval)
{
    poll_interval_ = interval;
}

std::vector<CallbackDispatcher::Stats> CallbackDispatcher::GetStatistics() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Stats> stats;
    stats.reserve(entries_.size());
    for (const auto& entry : entries_) {
        Stats s;
        s.name = entry.name;
        s.priority = entry.priority;
        s.total_calls = entry.total_calls;
        if (entry.total_calls > 0) {
            s.avg_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                entry.total_duration / entry.total_calls);
        } else {
            s.avg_duration = std::chrono::microseconds{0};
        }
        stats.push_back(s);
    }
    return stats;
}

void CallbackDispatcher::PrintStatistics() const
{
    auto stats = GetStatistics();
    for (const auto& s : stats) {
        spdlog::info("Stats - name: {}, priority: {}, total_calls: {}, avg_duration(us): {}",
                     s.name,
                     s.priority,
                     s.total_calls,
                     s.avg_duration.count());
    }
}

void CallbackDispatcher::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

size_t CallbackDispatcher::Size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

void CallbackDispatcher::RunLoop()
{
    while (running_.load()) {
        PollOnce();
        std::this_thread::sleep_for(poll_interval_);
    }
}

}  // namespace ms_slam::slam_common
