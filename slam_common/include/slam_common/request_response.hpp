/**
 * @file request_response.hpp
 * @brief 基于 iceoryx2 的请求响应封装
 */
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include <iox2/iceoryx2.hpp>

namespace ms_slam::slam_common
{
/**
 * @brief 请求响应服务器封装
 * @tparam RequestType 请求消息类型
 * @tparam ResponseType 响应消息类型
 */
template <typename RequestType, typename ResponseType>
class RPCServer
{
  public:
    /// 服务端处理回调类型
    using ServerCallback = std::function<ResponseType(const RequestType&)>;

    /**
     * @brief 构造请求响应服务器
     * @param node iceoryx 节点
     * @param service_name 服务名称
     */
    RPCServer(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name)
    : node_(node)
    , service_{node_->service_builder(iox2::ServiceName::create(service_name.c_str()).expect("valid service name"))
                       .request_response<RequestType, ResponseType>()
                       .open_or_create()
                       .expect("successful service open/create")}
    , server_{service_->server_builder().create().expect("successful server creation")}
    {
    }

    /**
     * @brief 接收并处理单个请求（非阻塞）
     * @return 成功处理返回 true
     */
    bool receive_and_respond()
    {
        auto active_request = server_->receive().expect("receive successful");
        if (active_request.has_value()) {
            if (callback_) {
                // 获取请求数据
                const RequestType& request = active_request->payload();

                // 调用回调处理
                ResponseType response_data = callback_(request);

                // 发送响应（使用复制接口）
                active_request->send_copy(response_data).expect("send successful");
            }
            return true;
        }
        return false;
    }

    /**
     * @brief 获取原始请求对象
     * @return ActiveRequest 对象
     */
    auto receive()
    {
        return server_->receive().expect("receive successful");
    }

    /**
     * @brief 设置请求处理回调
     * @param callback 处理回调
     */
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

/**
 * @brief 请求响应客户端封装
 * @tparam RequestType 请求消息类型
 * @tparam ResponseType 响应消息类型
 */
template <typename RequestType, typename ResponseType>
class RPCClient
{
  public:
    /**
     * @brief 构造请求响应客户端
     * @param node iceoryx 节点
     * @param service_name 服务名称
     */
    RPCClient(std::shared_ptr<iox2::Node<iox2::ServiceType::Ipc>> node, const std::string& service_name)
    : node_(node)
    , service_{node_->service_builder(iox2::ServiceName::create(service_name.c_str()).expect("valid service name"))
                       .request_response<RequestType, ResponseType>()
                       .open_or_create()
                       .expect("successful service open/create")}
    , client_{service_->client_builder().create().expect("successful client creation")}
    {
    }

    /**
     * @brief 发送请求并返回异步响应句柄
     * @param request_data 请求数据
     * @return PendingResponse 对象
     */
    auto send(const RequestType& request_data)
    {
        return client_->send_copy(request_data).expect("send successful");
    }

    /**
     * @brief 发送请求并阻塞等待响应
     * @param request_data 请求数据
     * @param timeout 等待超时时间
     * @return 成功返回响应，超时返回 nullopt
     */
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
