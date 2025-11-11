// CallbackDispatcher 测试示例
// 演示如何使用 CallbackDispatcher 统一管理多个 subscriber 和 server 的回调

#include <thread>
#include <chrono>
#include <atomic>
#include <random>

#include <spdlog/spdlog.h>
#include <flatbuffers/flatbuffers.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/Imu_generated.h>

#include <slam_common/callback_dispatcher.hpp>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>
#include <slam_common/request_response.hpp>

using namespace ms_slam::slam_common;

// ============================================================================
// 辅助函数：创建 Foxglove 消息
// ============================================================================

/// 创建 Foxglove IMU 数据
flatbuffers::DetachedBuffer create_foxglove_imu(uint32_t seq)
{
    flatbuffers::FlatBufferBuilder fbb(1024);

    // 创建时间戳
    uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
    uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    foxglove::Time timestamp(stamp_sec, stamp_nsec);

    // 创建 frame_id
    auto frame_id = fbb.CreateString("imu_frame");

    // 创建角速度和线性加速度
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

    auto angular_velocity = foxglove::CreateVector3(fbb, dist(gen), dist(gen), dist(gen));
    auto linear_acceleration = foxglove::CreateVector3(fbb, dist(gen) * 9.8, dist(gen) * 9.8, 9.8 + dist(gen));

    // 创建 Imu
    auto imu = foxglove::CreateImu(
        fbb,
        &timestamp,
        frame_id,
        angular_velocity,
        linear_acceleration
    );

    fbb.Finish(imu);
    return fbb.Release();
}

/// 创建 Foxglove PointCloud 数据
flatbuffers::DetachedBuffer create_foxglove_pointcloud(uint32_t seq, size_t num_points = 100)
{
    flatbuffers::FlatBufferBuilder fbb(1024 * 1024);

    // 创建时间戳
    uint32_t stamp_sec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() / 1000000000);
    uint32_t stamp_nsec = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count() % 1000000000);
    foxglove::Time timestamp(stamp_sec, stamp_nsec);

    // 创建 frame_id
    auto frame_id = fbb.CreateString("lidar_frame");

    // 创建点云字段 (x, y, z, intensity)
    std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fields;
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "x", 0, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "y", 4, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "z", 8, foxglove::NumericType_FLOAT32));
    fields.push_back(foxglove::CreatePackedElementFieldDirect(fbb, "intensity", 12, foxglove::NumericType_FLOAT32));
    auto fields_vector = fbb.CreateVector(fields);

    // 生成随机点云数据
    std::vector<uint8_t> point_data;
    point_data.reserve(num_points * 16);  // 4 fields * 4 bytes each

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<float> coord_dist(-10.0f, 10.0f);
    thread_local std::uniform_real_distribution<float> intensity_dist(0.0f, 100.0f);

    for (size_t i = 0; i < num_points; ++i) {
        float x = coord_dist(gen);
        float y = coord_dist(gen);
        float z = coord_dist(gen);
        float intensity = intensity_dist(gen);

        // 转换 float 为 bytes
        const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(&x);
        const uint8_t* y_bytes = reinterpret_cast<const uint8_t*>(&y);
        const uint8_t* z_bytes = reinterpret_cast<const uint8_t*>(&z);
        const uint8_t* intensity_bytes = reinterpret_cast<const uint8_t*>(&intensity);

        point_data.insert(point_data.end(), x_bytes, x_bytes + 4);
        point_data.insert(point_data.end(), y_bytes, y_bytes + 4);
        point_data.insert(point_data.end(), z_bytes, z_bytes + 4);
        point_data.insert(point_data.end(), intensity_bytes, intensity_bytes + 4);
    }

    auto data_vector = fbb.CreateVector(point_data);

    // 创建 PointCloud
    auto pointcloud = foxglove::CreatePointCloud(
        fbb,
        &timestamp,
        frame_id,
        0,  // pose (optional)
        16,  // point_stride (4 fields * 4 bytes)
        fields_vector,
        data_vector
    );

    fbb.Finish(pointcloud);
    return fbb.Release();
}

// ============================================================================
// RPC 测试数据类型
// ============================================================================

struct CalculateRequest
{
    static constexpr const char* IOX2_TYPE_NAME = "CalculateRequest";
    double value;
    int32_t operation;  // 0=square, 1=cube
};

struct CalculateResponse
{
    static constexpr const char* IOX2_TYPE_NAME = "CalculateResponse";
    double result;
    int32_t status;
};

// ============================================================================
// 示例 1：使用 CallbackDispatcher 管理多个 FBSSubscriber
// ============================================================================

void example_multiple_subscribers()
{
    spdlog::info("=== 示例 1:管理多个 FBSSubscriber ===");

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    auto imu_pub = std::make_shared<FBSPublisher<FoxgloveImu>>(node, "ImuTopic", PubSubConfig{.subscriber_max_buffer_size = 300});
    auto cloud_pub = std::make_shared<FBSPublisher<FoxglovePointCloud>>(node, "CloudTopic");

    // 创建 subscribers
    auto imu_sub = std::make_shared<FBSSubscriber<FoxgloveImu>>(node, "ImuTopic", nullptr, PubSubConfig{.subscriber_max_buffer_size = 300});
    auto cloud_sub = std::make_shared<FBSSubscriber<FoxglovePointCloud>>(node, "CloudTopic");

    // 创建 dispatcher
    CallbackDispatcher dispatcher;
    dispatcher.set_poll_interval(std::chrono::milliseconds(1));

    // 统计接收到的消息数量
    std::atomic<int> imu_count{0};
    std::atomic<int> cloud_count{0};
    imu_sub->set_receive_callback(
        [&imu_count](const FoxgloveImu& imu_wrapper) {
            imu_count++;
            const foxglove::Imu* imu = imu_wrapper.get();
            if (imu_count <= 3) {
                spdlog::info("IMU #{}: frame_id={}, angular_vel=({:.2f},{:.2f},{:.2f})",
                    imu_count.load(),
                    imu->frame_id()->c_str(),
                    imu->angular_velocity()->x(),
                    imu->angular_velocity()->y(),
                    imu->angular_velocity()->z());
            }
        }
    );

    cloud_sub->set_receive_callback(
        [&cloud_count](const FoxglovePointCloud& pc_wrapper) {
            cloud_count++;
            const foxglove::PointCloud* pc = pc_wrapper.get();
            size_t num_points = pc->data()->size() / pc->point_stride();
            spdlog::info("PointCloud #{}: frame_id={}, points={}",
                cloud_count.load(), pc->frame_id()->c_str(), num_points);
        }
    );

    // 注册 IMU subscriber（高优先级）
    dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(
        imu_sub,        "IMU_Subscriber",
        10  // 高优先级
    );

    // 注册点云 subscriber（低优先级）
    dispatcher.register_subscriber<FBSSubscriber<FoxglovePointCloud>>(
        cloud_sub,
        "Cloud_Subscriber",
        5  // 低优先级
    );

    // 启动 dispatcher
    dispatcher.start();

    // 发布数据
    std::thread pub_thread([&]() {
        for (int i = 0; i < 10; i++) {
            // 发布 IMU 数据（高频）
            auto imu_buffer = create_foxglove_imu(i);
            imu_pub->publish_raw(imu_buffer.data(), imu_buffer.size());

            // 每 5 次发布一次点云数据（低频）
            if (i % 5 == 0) {
                auto cloud_buffer = create_foxglove_pointcloud(i, 100);
                cloud_pub->publish_raw(cloud_buffer.data(), cloud_buffer.size());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    pub_thread.join();

    // 等待所有消息被处理
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 停止 dispatcher
    dispatcher.stop();

    // 打印统计信息
    dispatcher.print_statistics();

    spdlog::info("总接收: IMU={}, PointCloud={}", imu_count.load(), cloud_count.load());
    spdlog::info("✓ 示例 1 完成");
}

// ============================================================================
// 示例 2：使用 CallbackDispatcher 管理 Subscriber 和 RPC Server
// ============================================================================

void example_mixed_callbacks()
{
    spdlog::info("=== 示例 2:管理 FBSSubscriber + RPC Server ===");

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    // 创建 IMU pub-sub
    PubSubConfig config;
    config.subscriber_max_buffer_size = 300;
    auto imu_pub = std::make_shared<FBSPublisher<FoxgloveImu>>(node, "ImuTopic2", config);
    auto imu_sub = std::make_shared<FBSSubscriber<FoxgloveImu>>(node, "ImuTopic2", nullptr, config);

    // 创建 RPC server
    auto calc_server = std::make_shared<RPCServer<CalculateRequest, CalculateResponse>>(
        node, "CalcService");

    calc_server->set_callback([](const CalculateRequest& req) -> CalculateResponse {
        spdlog::info("RPC: 收到请求 value={}, op={}", req.value, req.operation);

        CalculateResponse resp;
        resp.status = 0;

        switch (req.operation) {
            case 0:  // square
                resp.result = req.value * req.value;
                break;
            case 1:  // cube
                resp.result = req.value * req.value * req.value;
                break;
            default:
                resp.status = -1;
                resp.result = 0.0;
        }

        spdlog::info("RPC: 发送响应 result={}", resp.result);
        return resp;
    });

    // 创建 RPC client
    auto calc_client = std::make_shared<RPCClient<CalculateRequest, CalculateResponse>>(
        node, "CalcService");

    // 创建 dispatcher
    CallbackDispatcher dispatcher;
    dispatcher.set_poll_interval(std::chrono::milliseconds(1));

    // 注册 subscriber
    std::atomic<int> imu_count{0};
    imu_sub->set_receive_callback(
        [&imu_count](const FoxgloveImu& imu_wrapper) {
            imu_count++;
            const foxglove::Imu* imu = imu_wrapper.get();
            spdlog::info("IMU #{}: frame_id={}", imu_count.load(), imu->frame_id()->c_str());
        }
    );

    dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(
        imu_sub,
        "IMU_Subscriber",
        5
    );

    // 注册 RPC server
    dispatcher.register_server(calc_server, "Calc_Server", 10);

    // 启动 dispatcher
    dispatcher.start();

    // 发布 IMU 数据和发送 RPC 请求
    std::thread work_thread([&]() {
        for (int i = 0; i < 5; i++) {
            // 发布 IMU
            auto imu_buffer = create_foxglove_imu(i);
            imu_pub->publish_raw(imu_buffer.data(), imu_buffer.size());

            // 发送 RPC 请求
            CalculateRequest req{static_cast<double>(i + 1), i % 2};
            auto resp = calc_client->send_and_wait(req, std::chrono::milliseconds(50));

            if (resp.has_value()) {
                spdlog::info("Client: 收到响应 result={}", resp->result);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    work_thread.join();

    // 等待处理完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 停止 dispatcher
    dispatcher.stop();

    // 打印统计信息
    dispatcher.print_statistics();

    spdlog::info("总接收 IMU: {}", imu_count.load());
    spdlog::info("✓ 示例 2 完成");
}

// ============================================================================
// 示例 3：动态注册和注销
// ============================================================================

void example_dynamic_registration()
{
    spdlog::info("=== 示例 3:动态注册和注销 ===");

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    PubSubConfig config;
    config.subscriber_max_buffer_size = 300;
    auto imu_pub = std::make_shared<FBSPublisher<FoxgloveImu>>(node, "ImuTopic3", config);
    auto imu_sub = std::make_shared<FBSSubscriber<FoxgloveImu>>(node, "ImuTopic3", nullptr, config);

    CallbackDispatcher dispatcher;
    dispatcher.set_poll_interval(std::chrono::milliseconds(1));

    std::atomic<int> count{0};
    imu_sub->set_receive_callback(
        [&count](const FoxgloveImu& imu_wrapper) {
            count++;
            const foxglove::Imu* imu = imu_wrapper.get();
            spdlog::info("IMU: count={}, frame_id={}", count.load(), imu->frame_id()->c_str());
        }
    );

    // 注册 subscriber
    uint64_t handle = dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(
        imu_sub,
        "IMU_Subscriber"
    );

    spdlog::info("注册 subscriber, handle={}", handle);
    spdlog::info("当前注册数量: {}", dispatcher.size());

    dispatcher.start();

    // 发布 5 条消息
    for (int i = 0; i < 5; i++) {
        auto imu_buffer = create_foxglove_imu(i);
        imu_pub->publish_raw(imu_buffer.data(), imu_buffer.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 注销 subscriber
    spdlog::info("注销 subscriber, handle={}", handle);
    dispatcher.unregister(handle);
    spdlog::info("当前注册数量: {}", dispatcher.size());

    // 继续发布消息（不会被接收）
    for (int i = 5; i < 10; i++) {
        auto imu_buffer = create_foxglove_imu(i);
        imu_pub->publish_raw(imu_buffer.data(), imu_buffer.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    dispatcher.stop();

    spdlog::info("总接收消息数: {} (应该是 5)", count.load());
    spdlog::info("✓ 示例 3 完成");
}

// ============================================================================
// 示例 4：手动轮询模式
// ============================================================================

void example_manual_polling()
{
    spdlog::info("=== 示例 4：手动轮询模式 ===");

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("node creation"));

    PubSubConfig config;
    auto imu_pub = std::make_shared<FBSPublisher<FoxgloveImu>>(node, "ImuTopic4", config);
    auto imu_sub = std::make_shared<FBSSubscriber<FoxgloveImu>>(node, "ImuTopic4", nullptr, config);

    CallbackDispatcher dispatcher;

    std::atomic<int> count{0};
    imu_sub->set_receive_callback(
        [&count](const FoxgloveImu& imu_wrapper) {
            count++;
            const foxglove::Imu* imu = imu_wrapper.get();
            spdlog::info("IMU: count={}, frame_id={}", count.load(), imu->frame_id()->c_str());
        }
    );

    dispatcher.register_subscriber<FBSSubscriber<FoxgloveImu>>(
        imu_sub,
        "IMU_Subscriber"
    );

    // 不启动自动线程，手动调用 poll_once()

    // 发布 5 条消息
    for (int i = 0; i < 5; i++) {
        auto imu_buffer = create_foxglove_imu(i);
        imu_pub->publish_raw(imu_buffer.data(), imu_buffer.size());

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 手动轮询
        dispatcher.poll_once();
    }

    dispatcher.print_statistics();

    spdlog::info("总接收消息数: {}", count.load());
    spdlog::info("✓ 示例 4 完成");
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    spdlog::info("CallbackDispatcher 测试");

    try {
        example_multiple_subscribers();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        example_mixed_callbacks();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        example_dynamic_registration();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        example_manual_polling();

        spdlog::info("所有测试完成！");

    } catch (const std::exception& e) {
        spdlog::error("错误: {}", e.what());
        return 1;
    }

    return 0;
}
