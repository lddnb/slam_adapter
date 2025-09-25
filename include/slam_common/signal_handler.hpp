#pragma once

#include <sys/wait.h>
#include <iostream>
#include <csignal>
#include <cstring>

#include <cpptrace/cpptrace.hpp>

namespace ms_slam::slam_common{

/**
 * @brief 在信号处理器中被调用的函数，用于启动堆栈追踪程序
 * 
 * @param buffer 
 * @param count 
 */
void do_signal_safe_trace(cpptrace::frame_ptr* buffer, std::size_t count) {
    // 创建管道和子进程
    int pipe_fd[2];;
    if (pipe(pipe_fd) == -1) {
        return;
    }

    const pid_t pid = fork();
    if (pid == -1) {
        const char* fork_failure_message = "fork() failed\n";
        write(STDERR_FILENO, fork_failure_message, strlen(fork_failure_message));
        return;
    }

    if (pid == 0) {
        // 子进程
        // 将管道的读端重定向到标准输入，然后执行追踪程序
        dup2(pipe_fd[0], STDIN_FILENO);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        // 启动追踪程序
        execl("./signal_tracer", "signal_tracer", nullptr);
        // 如果 execl 失败，检查二进制文件是否在同一目录下
        const char* exec_failure_message = "exec(tracer) failed\n";
        write(STDERR_FILENO, exec_failure_message, strlen(exec_failure_message));
        _exit(1);
    }

    // 父进程
    // 将堆栈信息写入管道
    close(pipe_fd[0]);
    for (std::size_t i = 0; i < count; i++) {
        cpptrace::safe_object_frame frame;
        cpptrace::get_safe_object_frame(buffer[i], &frame);
        write(pipe_fd[1], &frame, sizeof(frame));
    }
    close(pipe_fd[1]);

    // 等待子进程结束
    waitpid(pid, nullptr, 0);
}

/**
 * @brief 信号处理器
 * 
 * @param signo 
 * @param info 
 * @param context 
 */
void signal_handler(int signo, siginfo_t*, void*) {
    const char* message;
    switch (signo) {
        case SIGSEGV:
            message = "FATAL: Segmentation fault (SIGSEGV)\n";
            break;
        case SIGABRT:
            message = "FATAL: Abort (SIGABRT)\n";
            break;
        case SIGFPE:
            message = "FATAL: Floating point exception (SIGFPE)\n";
            break;
        default:
            message = "FATAL: Unexpected signal\n";
            break;
    }
    write(STDERR_FILENO, message, strlen(message));

    // 获取原始堆栈信息
    constexpr std::size_t max_frames = 100;
    cpptrace::frame_ptr buffer[max_frames];
    std::size_t frame_count = cpptrace::safe_generate_raw_trace(buffer, max_frames);

    // 调用追踪函数
    do_signal_safe_trace(buffer, frame_count);

    // 单元测试时直接返回1，而不是让程序崩溃异常退出，从而通过单元测试
#ifdef BUILD_TESTS
    _exit(1);
#endif

    // 恢复默认处理并重新引发信号，以便生成 coredump (如果系统配置允许)
    signal(signo, SIG_DFL);
    raise(signo);
}

/**
 * @brief cpptrace 依赖检查和预热
 * 
 */
void warmup_cpptrace() {
    if (!cpptrace::can_signal_safe_unwind() || !cpptrace::can_get_safe_object_frame()) {
        std::cerr << "Signal-safe tracing not supported on this system" << std::endl;
        return;
    }
    cpptrace::frame_ptr buffer[1];
    cpptrace::safe_generate_raw_trace(buffer, 1);
    cpptrace::safe_object_frame frame;
    cpptrace::get_safe_object_frame(buffer[0], &frame);
}

/**
 * @brief 设置信号处理器
 * 
 */
void InstallFailureSignalHandler() {
    struct sigaction action = {};
    action.sa_sigaction = &signal_handler;
    action.sa_flags = SA_RESETHAND | SA_SIGINFO;

    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
}

} // namespace