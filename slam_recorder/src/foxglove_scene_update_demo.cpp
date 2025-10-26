#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>

#include <flatbuffers/flatbuffers.h>
#include <fbs/SceneUpdate_generated.h>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

using namespace std::chrono_literals;
using ms_slam::slam_common::FBSPublisher;
using ms_slam::slam_common::FoxgloveSceneUpdate;

namespace {

constexpr const char* kSceneTopic = "/marker";
std::atomic_bool g_running{true};

void handleSignal(int) {
    g_running.store(false);
}

foxglove::Time makeTimestamp(const std::chrono::system_clock::time_point& tp) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch());
    const auto sec = static_cast<uint32_t>(ns.count() / 1'000'000'000LL);
    const auto nsec = static_cast<uint32_t>(ns.count() % 1'000'000'000LL);
    return foxglove::Time(sec, nsec);
}

}  // namespace

int main() {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("Failed to create iceoryx2 node"));
    FBSPublisher<FoxgloveSceneUpdate> publisher(node, kSceneTopic);

    std::cout << "SceneUpdate 椭球演示开始，发布服务: " << kSceneTopic << std::endl;

    flatbuffers::FlatBufferBuilder builder(1024);
    const auto start_time = std::chrono::steady_clock::now();

    while (g_running.load()) {
        builder.Clear();

        const auto now_system = std::chrono::system_clock::now();
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

        constexpr double trajectory_radius = 1.5;
        const double center_x = trajectory_radius * std::cos(t);
        const double center_y = trajectory_radius * std::sin(t);
        const double center_z = 0.5 * std::sin(t * 0.5);

        const double base_scale = 0.4 + 0.1 * std::sin(t * 0.7);
        const double size_x = base_scale * (1.0 + 0.5 * std::sin(t * 0.9));
        const double size_y = base_scale * (1.5 + 0.3 * std::cos(t * 1.2));
        const double size_z = base_scale * (0.7 + 0.4 * std::sin(t * 1.5));

        const double r = 0.3 + 0.7 * std::fabs(std::sin(t));
        const double g = 0.2 + 0.6 * std::fabs(std::cos(t * 0.6));
        const double b = 0.4 + 0.5 * std::fabs(std::sin(t * 1.3));

        const auto position_offset = foxglove::CreateVector3(builder, center_x, center_y, center_z);
        const auto orientation_offset = foxglove::CreateQuaternion(builder, 0.0, 0.0, 0.0, 1.0);
        const auto pose_offset = foxglove::CreatePose(builder, position_offset, orientation_offset);
        const auto size_offset = foxglove::CreateVector3(builder, size_x, size_y, size_z);
        const auto color_offset = foxglove::CreateColor(builder, r, g, b, 0.8);
        const auto sphere_offset = foxglove::CreateSpherePrimitive(builder, pose_offset, size_offset, color_offset);
        const auto spheres_offset = builder.CreateVector(&sphere_offset, 1);

        const auto frame_id_offset = builder.CreateString("map");
        const auto entity_id_offset = builder.CreateString("moving_ellipsoid");
        const auto timestamp = makeTimestamp(now_system);

        foxglove::SceneEntityBuilder entity_builder(builder);
        entity_builder.add_timestamp(&timestamp);
        entity_builder.add_frame_id(frame_id_offset);
        entity_builder.add_id(entity_id_offset);
        entity_builder.add_spheres(spheres_offset);
        const auto entity_offset = entity_builder.Finish();

        const auto entities_offset = builder.CreateVector(&entity_offset, 1);

        foxglove::SceneUpdateBuilder update_builder(builder);
        update_builder.add_entities(entities_offset);
        const auto update_offset = update_builder.Finish();

        foxglove::FinishSceneUpdateBuffer(builder, update_offset);

        if (!publisher.publish_from_builder(builder)) {
            std::cerr << "发布 SceneUpdate 失败" << std::endl;
        }

        std::cout << "publish at position: " <<  center_x << ", " << center_y << ", " << center_z << std::endl;
        std::this_thread::sleep_for(500ms);
    }

    std::cout << "SceneUpdate 演示结束" << std::endl;
    return 0;
}
