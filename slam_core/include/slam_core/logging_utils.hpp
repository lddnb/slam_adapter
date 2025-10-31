/**
 * @file logging_utils.hpp
 * @brief 定义日志相关的宏
 */

#pragma once

#include <atomic>
#include <spdlog/spdlog.h>
#include <Eigen/Core>

template <class Derived>
struct EigenWrap {
    const Eigen::MatrixBase<Derived>& m;
    Eigen::IOFormat fmt;
};

template <class Derived>
EigenWrap<Derived> as_eigen(const Eigen::MatrixBase<Derived>& m,
                            Eigen::IOFormat io = Eigen::IOFormat(
                                Eigen::StreamPrecision, 0, ", ", "\n", "[", "]")) {
    return {m, io};
}

template <class Derived>
struct fmt::formatter<EigenWrap<Derived>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <class FormatContext>
    auto format(const EigenWrap<Derived>& w, FormatContext& ctx) {
        std::ostringstream oss;
        oss << w.m.derived().format(w.fmt);
        return fmt::format_to(ctx.out(), "{}", oss.str());
    }
};

// --- 拼接宏，保证调用点唯一名 ---
#define SPD_CAT_INNER(a, b) a##b
#define SPD_CAT(a, b) SPD_CAT_INNER(a, b)
#define SPD_UNIQUE_PER_LINE(base) SPD_CAT(base, __LINE__)

// ----------------------------
// CHECK
// ----------------------------
[[noreturn]] inline void check_failed(const char* condition, const char* file, int line)
{
    spdlog::critical("CHECK FAILED: {} at {}:{}", condition, file, line);
    std::terminate();
}

#define CHECK(condition)                              \
    if (!(condition)) {                               \
        check_failed(#condition, __FILE__, __LINE__); \
    }

// ----------------------------
// LOG_FIRST_N
// ----------------------------
template <std::size_t N, typename... Args>
inline void log_first_n_impl(std::atomic<std::size_t>& counter, spdlog::level::level_enum lvl, fmt::format_string<Args...> fmt, Args&&... args)
{
    static_assert(N > 0, "N must be > 0");
    auto* lg = spdlog::default_logger_raw();
    if (!lg || !lg->should_log(lvl)) return;

    const std::size_t cur = counter.fetch_add(1, std::memory_order_relaxed);
    if (cur < N) {
        SPDLOG_LOGGER_CALL(lg, lvl, fmt, std::forward<Args>(args)...);
    }
}

#define LOG_FIRST_N(lvl, N, fmt, ...)                                                                          \
    do {                                                                                                       \
        static std::atomic<std::size_t> SPD_UNIQUE_PER_LINE(_spd_cnt_firstn_){0};                              \
        ::log_first_n_impl<(N)>(SPD_UNIQUE_PER_LINE(_spd_cnt_firstn_), (lvl), FMT_STRING(fmt), ##__VA_ARGS__); \
    } while (0)

// ----------------------------
// LOG_EVERY_N
// ----------------------------
template <std::size_t N, typename... Args>
inline void log_every_n_impl(std::atomic<std::size_t>& counter, spdlog::level::level_enum lvl, fmt::format_string<Args...> fmt, Args&&... args)
{
    static_assert(N > 0, "N must be > 0");
    auto* lg = spdlog::default_logger_raw();
    if (!lg || !lg->should_log(lvl)) return;

    const std::size_t cur = counter.fetch_add(1, std::memory_order_relaxed);
    if constexpr ((N & (N - 1)) == 0) {  // N 为 2 的幂时的微优化
        if ((cur & (N - 1)) == 0) {
            SPDLOG_LOGGER_CALL(lg, lvl, fmt, std::forward<Args>(args)...);
        }
    } else {
        if ((cur % N) == 0) {
            SPDLOG_LOGGER_CALL(lg, lvl, fmt, std::forward<Args>(args)...);
        }
    }
}

#define LOG_EVERY_N(lvl, N, fmt, ...)                                                                          \
    do {                                                                                                       \
        static std::atomic<std::size_t> SPD_UNIQUE_PER_LINE(_spd_cnt_everyn_){0};                              \
        ::log_every_n_impl<(N)>(SPD_UNIQUE_PER_LINE(_spd_cnt_everyn_), (lvl), FMT_STRING(fmt), ##__VA_ARGS__); \
    } while (0)

// ----------------------------
// LOG_IF
// ----------------------------
template <typename... Args>
inline void log_if_impl(bool cond, spdlog::level::level_enum lvl, fmt::format_string<Args...> fmt, Args&&... args)
{
    if (!cond) return;
    auto* lg = spdlog::default_logger_raw();
    if (!lg || !lg->should_log(lvl)) return;
    SPDLOG_LOGGER_CALL(lg, lvl, fmt, std::forward<Args>(args)...);
}

#define LOG_IF(lvl, cond, fmt, ...)                                                     \
    do {                                                                                \
        ::log_if_impl((cond), (lvl), FMT_STRING(fmt), ##__VA_ARGS__);                   \
    } while (0)

using spdlog::level::debug;
using spdlog::level::info;
using spdlog::level::warn;
using spdlog::level::err;
using spdlog::level::critical;