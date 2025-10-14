#pragma once

#include <memory>
#include <string>
#include <functional>

#include <iox2/iceoryx2.hpp>

namespace ms_slam::slam_common
{

// ============================================================================
// RPCServer: 简单封装 iceoryx2 的 request-response server
// ============================================================================

template <typename RequestType, typename ResponseType>
class RPCServer
{
  public:
    using ServerCallback = std::function<ResponseType(const RequestType&)>;

    RPCServer(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name)
    : node_(node)
    , service_{node_->service_builder(iox2::ServiceName::create(service_name.c_str()).expect("valid service name"))
                       .request_response<RequestType, ResponseType>()
                       .open_or_create()
                       .expect("successful service open/create")}
    , server_{service_->server_builder().create().expect("successful server creation")}
    {
    }

    // 接收并处理一个请求（非阻塞）
    // 如果设置了 callback，会自动调用并发送响应
    // 返回 true 表示处理了一个请求，false 表示没有待处理的请求
    bool receive_and_respond()
    {
        auto active_request = server_->receive().expect("receive successful");
        if (active_request.has_value()) {
            if (callback_) {
                // 获取请求数据
                const RequestType& request = active_request->payload();

                // 调用回调处理
                ResponseType response_data = callback_(request);

                // 发送响应（使用简单的 copy API）
                active_request->send_copy(response_data).expect("send successful");
            }
            return true;
        }
        return false;
    }

    // 原始接收方法（返回 ActiveRequest，用户自己处理响应）
    auto receive()
    {
        return server_->receive().expect("receive successful");
    }

    // 设置请求处理回调
    void set_callback(ServerCallback callback)
    {
        callback_ = callback;
    }

  private:
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;

    iox::optional<iox2::PortFactoryRequestResponse<iox2::ServiceType::Ipc, RequestType, void, ResponseType, void>> service_;
    iox::optional<iox2::Server<iox2::ServiceType::Ipc, RequestType, void, ResponseType, void>> server_;

    ServerCallback callback_;
};

// ============================================================================
// RPCClient: 简单封装 iceoryx2 的 request-response client
// ============================================================================

template <typename RequestType, typename ResponseType>
class RPCClient
{
  public:
    RPCClient(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name)
    : node_(node)
    , service_{node_->service_builder(iox2::ServiceName::create(service_name.c_str()).expect("valid service name"))
                       .request_response<RequestType, ResponseType>()
                       .open_or_create()
                       .expect("successful service open/create")}
    , client_{service_->client_builder().create().expect("successful client creation")}
    {
    }

    // 发送请求并返回 PendingResponse（用于接收响应）
    auto send(const RequestType& request_data)
    {
        return client_->send_copy(request_data).expect("send successful");
    }

    // 发送请求并等待单个响应（阻塞，带超时）
    iox::optional<ResponseType> send_and_wait(const RequestType& request_data, std::chrono::milliseconds timeout = std::chrono::milliseconds(100))
    {
        auto pending = send(request_data);

        auto start = std::chrono::steady_clock::now();
        while (true) {
            auto response = pending.receive().expect("receive successful");
            if (response.has_value()) {
                return response->payload();
            }

            // 检查超时
            if (std::chrono::steady_clock::now() - start >= timeout) {
                return iox::nullopt;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

  private:
    std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;

    iox::optional<iox2::PortFactoryRequestResponse<iox2::ServiceType::Ipc, RequestType, void, ResponseType, void>> service_;
    iox::optional<iox2::Client<iox2::ServiceType::Ipc, RequestType, void, ResponseType, void>> client_;
};

}  // namespace ms_slam::slam_common
