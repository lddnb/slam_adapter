// RPC (Request-Response) 通信测试示例
// 演示如何使用 RPCServer 和 RPCClient

#include <iostream>
#include <thread>
#include <chrono>

#include <slam_common/request_response.hpp>

using namespace ms_slam::slam_common;

// ============================================================================
// 定义请求和响应数据结构（简单的 POD 类型）
// ============================================================================

struct CalculateRequest
{
    static constexpr const char* IOX2_TYPE_NAME = "CalculateRequest";
    double value;
    int32_t operation;  // 0=square, 1=cube, 2=double
};

struct CalculateResponse
{
    static constexpr const char* IOX2_TYPE_NAME = "CalculateResponse";
    double result;
    int32_t status;  // 0=success, -1=error
};

// 打印辅助函数
inline std::ostream& operator<<(std::ostream& os, const CalculateRequest& req)
{
    const char* ops[] = {"square", "cube", "double"};
    os << "Request{value=" << req.value << ", op=" << ops[req.operation] << "}";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const CalculateResponse& resp)
{
    os << "Response{result=" << resp.result << ", status=" << resp.status << "}";
    return os;
}

// ============================================================================
// 示例 1：简单的 RPC（使用回调）
// ============================================================================

void example_simple_rpc()
{
    std::cout << "\n=== 示例 1：简单 RPC（使用回调）===\n" << std::endl;

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    // 创建 Server 并设置回调
    RPCServer<CalculateRequest, CalculateResponse> server(node, "CalculatorService");

    server.set_callback([](const CalculateRequest& req) -> CalculateResponse {
        std::cout << "Server: 收到 " << req << std::endl;

        CalculateResponse resp;
        resp.status = 0;

        switch (req.operation) {
            case 0:  // square
                resp.result = req.value * req.value;
                break;
            case 1:  // cube
                resp.result = req.value * req.value * req.value;
                break;
            case 2:  // double
                resp.result = req.value * 2.0;
                break;
            default:
                resp.status = -1;
                resp.result = 0.0;
        }

        std::cout << "Server: 发送 " << resp << std::endl;
        return resp;
    });

    // 创建 Client
    RPCClient<CalculateRequest, CalculateResponse> client(node, "CalculatorService");

    // Server 处理循环（在后台线程）
    std::atomic<bool> running{true};
    std::thread server_thread([&]() {
        while (running) {
            server.receive_and_respond();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Client 发送请求
    std::cout << "\nClient: 测试 5 的平方" << std::endl;
    CalculateRequest req1{5.0, 0};
    auto resp1 = client.send_and_wait(req1);
    if (resp1.has_value()) {
        std::cout << "Client: 收到 " << *resp1 << " (期望: 25)" << std::endl;
    }

    std::cout << "\nClient: 测试 3 的立方" << std::endl;
    CalculateRequest req2{3.0, 1};
    auto resp2 = client.send_and_wait(req2);
    if (resp2.has_value()) {
        std::cout << "Client: 收到 " << *resp2 << " (期望: 27)" << std::endl;
    }

    running = false;
    server_thread.join();

    std::cout << "\n✓ 示例 1 完成\n" << std::endl;
}

// ============================================================================
// 示例 2：手动处理请求（不使用回调）
// ============================================================================

void example_manual_handling()
{
    std::cout << "\n=== 示例 2：手动处理请求 ===\n" << std::endl;

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    // 创建 Server（不设置回调）
    RPCServer<CalculateRequest, CalculateResponse> server(node, "ManualService");

    // 创建 Client
    RPCClient<CalculateRequest, CalculateResponse> client(node, "ManualService");

    // Server 手动处理
    std::atomic<bool> running{true};
    std::thread server_thread([&]() {
        while (running) {
            auto active_request = server.receive();
            if (active_request.has_value()) {
                const CalculateRequest& req = active_request->payload();
                std::cout << "Server: 手动处理 " << req << std::endl;

                CalculateResponse resp;
                resp.result = req.value * 10;  // 简单乘以 10
                resp.status = 0;

                // 手动发送响应
                active_request->send_copy(resp).expect("send successful");
                std::cout << "Server: 手动发送 " << resp << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Client 发送请求
    std::cout << "\nClient: 发送 7 * 10" << std::endl;
    CalculateRequest req{7.0, 0};
    auto resp = client.send_and_wait(req);
    if (resp.has_value()) {
        std::cout << "Client: 收到 " << *resp << " (期望: 70)" << std::endl;
    }

    running = false;
    server_thread.join();

    std::cout << "\n✓ 示例 2 完成\n" << std::endl;
}

// ============================================================================
// 示例 3：异步请求（手动轮询响应）
// ============================================================================

void example_async_request()
{
    std::cout << "\n=== 示例 3：异步请求 ===\n" << std::endl;

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    // 创建 Server
    RPCServer<CalculateRequest, CalculateResponse> server(node, "AsyncService");
    server.set_callback([](const CalculateRequest& req) -> CalculateResponse {
        // 模拟耗时处理
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return {req.value * 100, 0};
    });

    // Server 处理循环
    std::atomic<bool> running{true};
    std::thread server_thread([&]() {
        while (running) {
            server.receive_and_respond();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // 创建 Client
    RPCClient<CalculateRequest, CalculateResponse> client(node, "AsyncService");

    // 异步发送请求
    std::cout << "Client: 发送异步请求..." << std::endl;
    CalculateRequest req{3.0, 0};
    auto pending = client.send(req);

    std::cout << "Client: 继续做其他工作..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "Client: 完成其他工作，现在等待响应..." << std::endl;

    // 手动轮询响应
    int retry = 0;
    while (retry < 20) {
        auto response = pending.receive().expect("receive successful");
        if (response.has_value()) {
            const CalculateResponse& resp = response->payload();
            std::cout << "Client: 收到异步响应 " << resp << " (期望: 300)" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        retry++;
    }

    running = false;
    server_thread.join();

    std::cout << "\n✓ 示例 3 完成\n" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    std::cout << "RPC (Request-Response) 通信测试\n" << std::endl;

    try {
        example_simple_rpc();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        example_manual_handling();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        example_async_request();

        std::cout << "\n所有测试完成！" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
