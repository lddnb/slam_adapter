// Copyright (c) 2024 Contributors to the Eclipse Foundation
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

#include "generic_publisher_subscriber.hpp"
#include "data_types.hpp"

#include <sys/wait.h>
#include <iostream>
#include <random>
#include <csignal>

#include <spdlog/spdlog.h>
#include <cpptrace/cpptrace.hpp>

// 在信号处理器中被调用的函数，用于启动追踪程序
void do_signal_safe_trace(cpptrace::frame_ptr* buffer, std::size_t count) {
    // 1. 创建管道和子进程
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

    if (pid == 0) { // 子进程
        // 2. 将管道的读端重定向到标准输入，然后执行追踪程序
        dup2(pipe_fd[0], STDIN_FILENO);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        // "tracer" 是我们编译的追踪程序的可执行文件名
        execl("./signal_tracer", "signal_tracer", nullptr);
        // 如果 execl 失败
        const char* exec_failure_message = "exec(tracer) failed: 确保 tracer 可执行文件在当前目录\n";
        write(STDERR_FILENO, exec_failure_message, strlen(exec_failure_message));
        _exit(1);
    }

    // 父进程
    // 3. 将堆栈信息写入管道
    close(pipe_fd[0]);
    for (std::size_t i = 0; i < count; i++) {
        cpptrace::safe_object_frame frame;
        cpptrace::get_safe_object_frame(buffer[i], &frame);
        write(pipe_fd[1], &frame, sizeof(frame));
    }
    close(pipe_fd[1]);

    // 4. 等待子进程结束
    waitpid(pid, nullptr, 0);
}

// 信号处理器
void signal_handler(int signo, siginfo_t* info, void* context) {
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
    
    // 恢复默认处理并重新引发信号，以便生成 coredump (如果系统配置允许)
    signal(signo, SIG_DFL);
    raise(signo);
}

// 预热 cpptrace 函数，确保在信号处理器中不会发生动态加载
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

// 安装信号处理器
void InstallFailureSignalHandler() {
    struct sigaction action = {};
    action.sa_sigaction = &signal_handler;
    action.sa_flags = SA_RESETHAND | SA_SIGINFO;

    // 注册我们关心的信号
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
}

// Generator function for PointCloud messages
PointCloud generate_point_cloud(uint32_t seq) {
    PointCloud cloud;
    cloud.seq = seq;
    cloud.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Generate some random points
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    cloud.points.clear();
    uint32_t num_points = 10 + (seq % 100); // 100-200 points
    cloud.num_points = num_points;
    std::cout << "num_points: " << num_points << std::endl;

    for (uint32_t i = 0; i < num_points; ++i) {
        PointCloudPoint point;
        point.tov = cloud.timestamp;
        point.x = coord_dist(gen);
        point.y = coord_dist(gen);
        point.z = coord_dist(gen);
        point.intensity = intensity_dist(gen);
        point.return_order = i % 3; // 0, 1, or 2
        cloud.points.push_back(point);
    }

    return cloud;
}

// Generator function for Image messages
Image generate_image(uint32_t seq) {
    Image image;
    image.seq = seq;
    image.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Set image format
    image.format.width = 640;
    image.format.height = 480;
    image.format.stride = image.format.width * 3; // RGB
    image.format.pixel_format[0] = 'R';
    image.format.pixel_format[1] = 'G';
    image.format.pixel_format[2] = 'B';
    image.format.pixel_format[3] = '8';

    // Generate simple pattern
    image.data.clear();
    for (uint32_t y = 0; y < image.format.height; ++y) {
        for (uint32_t x = 0; x < image.format.width; ++x) {
            uint8_t r = static_cast<uint8_t>((x + seq) % 256);
            uint8_t g = static_cast<uint8_t>((y + seq) % 256);
            uint8_t b = static_cast<uint8_t>((x + y + seq) % 256);

            image.data.push_back(r);
            image.data.push_back(g);
            image.data.push_back(b);
        }
    }

    return image;
}

int main() {
    using namespace iox2;

    warmup_cpptrace();
    InstallFailureSignalHandler();

    // Initialize node
    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        NodeBuilder().create<ServiceType::Ipc>().expect("successful node creation"));

    std::cout << "Creating generic publishers and subscribers..." << std::endl;

    // Create PointCloud publisher with generator function
    PointCloudPublisher pc_publisher(
        node,
        "Generic/PointCloud/Service",
        generate_point_cloud,
        std::chrono::milliseconds(500) // Publish every 500ms
    );

    // Create PointCloud subscriber with callback
    PointCloudSubscriber pc_subscriber(
        node,
        "Generic/PointCloud/Service",
        [](const PointCloud& cloud) {
            std::cout << "Received PointCloud: seq=" << cloud.seq
                      << ", points=" << cloud.points.size() << std::endl;
        }
    );

    // Create Image publisher with generator function
    ImagePublisher img_publisher(
        node,
        "Generic/Image/Service",
        generate_image,
        std::chrono::milliseconds(1000) // Publish every 1000ms
    );

    // Create Image subscriber with callback
    ImageSubscriber img_subscriber(
        node,
        "Generic/Image/Service",
        [](const Image& image) {
            std::cout << "Received Image: seq=" << image.seq
                      << ", size=" << image.format.width << "x" << image.format.height
                      << ", data_size=" << image.data.size() << std::endl;
        }
    );

    // Set additional callbacks for monitoring
    pc_publisher.set_publish_callback([](uint32_t seq, const PointCloud& cloud) {
        std::cout << "Published PointCloud " << seq << " with " << cloud.points.size() << " points" << std::endl;
    });

    img_publisher.set_publish_callback([](uint32_t seq, const Image& image) {
        std::cout << "Published Image " << seq << " (" << image.format.width << "x" << image.format.height << ")" << std::endl;
    });

    // Start all components
    std::cout << "Starting publishers and subscribers..." << std::endl;
    pc_publisher.start();
    pc_subscriber.start();
    img_publisher.start();
    img_subscriber.start();

    // Run for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Stop all components
    std::cout << "Stopping all components..." << std::endl;
    pc_publisher.stop();
    pc_subscriber.stop();
    img_publisher.stop();
    img_subscriber.stop();

    // Print statistics
    std::cout << "Statistics:" << std::endl;
    std::cout << "  PointCloud: Published=" << pc_publisher.get_published_count()
              << ", Received=" << pc_subscriber.get_received_count() << std::endl;
    std::cout << "  Image: Published=" << img_publisher.get_published_count()
              << ", Received=" << img_subscriber.get_received_count() << std::endl;

    // 下面是一个故意制造崩溃的例子
    // int* ptr = nullptr;
    // *ptr = 42; // 这将会触发 SIGSEGV

    // Demonstrate single-shot publishing/receiving
    std::cout << "\nDemonstrating single-shot operations..." << std::endl;

    auto single_cloud = generate_point_cloud(10);
    std::cout << "single_cloud: " << single_cloud.num_points << std::endl;
    std::cout << "Publisher state before publish: " 
              << (pc_publisher.is_running() ? "running" : "stopped") << std::endl;
    bool success = pc_publisher.publish(single_cloud);
    std::cout << "Single publish result: " << (success ? "success" : "failed") << std::endl;

    // Try to receive single message
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto received_cloud = pc_subscriber.receive_once();
    if (received_cloud.has_value()) {
        std::cout << "Single receive: got cloud with " << received_cloud->points.size() << " points" << std::endl;
    } else {
        std::cout << "Single receive: no message available" << std::endl;
    }

    std::cout << "Demo completed!" << std::endl;
    return 0;
}