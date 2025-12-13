/**
 * @file logging_utils.hpp
 * @brief 定义日志相关的宏
 */

#pragma once

#include <atomic>
#include <spdlog/spdlog.h>
#include <Eigen/Core>

/**
 * @brief Eigen矩阵的轻量包装器，配合fmt实现高性能日志输出
 * @tparam Derived Eigen矩阵派生类型
 */
template <class Derived>
struct EigenWrap {
    const Eigen::MatrixBase<Derived>& matrix;  ///< 需格式化的矩阵引用
    Eigen::IOFormat fmt;                      ///< 输出格式配置
};

/**
 * @brief 将Eigen矩阵封装为EigenWrap以便fmt格式化
 * @tparam Derived Eigen矩阵派生类型
 * @param matrix 待输出的矩阵
 * @param io 自定义输出格式，默认与Eigen标准输出一致
 * @return EigenWrap<Derived> 可直接用于spdlog/fmt的对象
 */
template <class Derived>
EigenWrap<Derived> as_eigen(
    const Eigen::MatrixBase<Derived>& matrix,
    Eigen::IOFormat io = Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]"))
{
    return {matrix, io};
}

template <class Derived>
struct fmt::formatter<EigenWrap<Derived>> {
    constexpr auto parse(format_parse_context& ctx)
    {
        auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') {
            throw fmt::format_error("invalid format for EigenWrap");
        }
        return it;
    }

    template <class FormatContext>
    auto format(const EigenWrap<Derived>& wrapper, FormatContext& ctx)
    {
        auto out = ctx.out();
        const auto rows = wrapper.matrix.rows();
        const auto cols = wrapper.matrix.cols();
        const auto& fmt_cfg = wrapper.fmt;
        const bool use_stream_precision = fmt_cfg.precision == Eigen::StreamPrecision;

        auto append_literal = [&](const std::string& literal) {
            if (!literal.empty()) {
                out = fmt::format_to(out, "{}", literal);
            }
        };

        append_literal(fmt_cfg.matPrefix);
        for (Eigen::Index r = 0; r < rows; ++r) {
            append_literal(fmt_cfg.rowPrefix);
            for (Eigen::Index c = 0; c < cols; ++c) {
                if (c > 0) {
                    append_literal(fmt_cfg.coeffSeparator);
                }
                const auto coeff = wrapper.matrix.coeff(r, c);
                if (use_stream_precision) {
                    out = fmt::format_to(out, "{}", coeff);
                } else {
                    out = fmt::format_to(out, "{:.{}f}", coeff, fmt_cfg.precision);
                }
            }
            append_literal(fmt_cfg.rowSuffix);
            if (r + 1 < rows) {
                append_literal(fmt_cfg.rowSeparator);
            }
        }
        append_literal(fmt_cfg.matSuffix);
        return out;
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
