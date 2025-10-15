/**
 * @file ros_read_pub.cpp
 * @brief ROS1 MCAP 转换并通过 iceoryx2 发布示例
 */
#include "slam_recorder/ros1_msg.hpp"

#include <iostream>
#include <cstring>
#include <vector>
#include <map>

#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <flatbuffers/flatbuffers.h>
#include <fbs/PointCloud_generated.h>
#include <fbs/CompressedImage_generated.h>
#include <fbs/Time_generated.h>
#include <fbs/PackedElementField_generated.h>
#include <fbs/Pose_generated.h>
#include <fbs/Vector3_generated.h>
#include <fbs/Quaternion_generated.h>
#include <fbs/Imu_generated.h>
#include <slam_common/flatbuffers_pub_sub.hpp>
#include <slam_common/foxglove_messages.hpp>

using namespace ms_slam::slam_common;

/**
 * @brief 将 ROS1 编码的 MCAP 文件转换并发布为 FlatBuffers
 * @param input_file 输入 MCAP 文件路径
 */
void convert_ros1_to_flatbuffers(const std::string& input_file) {
    std::cout << "=== 转换ROS1 MCAP到Flatbuffers MCAP ===" << std::endl;
    std::cout << "输入文件: " << input_file << std::endl;

    auto node = std::make_shared<iox2::Node<iox2::ServiceType::Ipc>>(
        iox2::NodeBuilder().create<iox2::ServiceType::Ipc>().expect("successful node creation"));
    FBSPublisher<FoxgloveCompressedImage> img_publisher(node, "/camera/image_raw");
    FBSPublisher<FoxglovePointCloud> pc_publisher(node, "/lidar_points");
    FBSPublisher<FoxgloveImu> imu_publisher(node, "/lidar_imu", PubSubConfig{.subscriber_max_buffer_size = 100});

    // 打开输入文件
    mcap::McapReader reader;
    auto status = reader.open(input_file);
    if (!status.ok()) {
        std::cerr << "无法打开输入文件: " << status.message << std::endl;
        return;
    }

    std::cout << "\n正在读取输入文件..." << std::endl;

    // 统计信息
    size_t pointcloud_count = 0;
    size_t image_count = 0;
    size_t imu_count = 0;
    size_t other_count = 0;

    // Channel ID映射（ROS topic -> MCAP channel ID）
    std::map<std::string, uint16_t> channel_map;
    uint16_t next_channel_id = 1;

    // 读取所有消息并转换
    auto messageView = reader.readMessages();

    std::optional<uint64_t> last_publish_time_ns;

    flatbuffers::FlatBufferBuilder fbb(1024 * 1024); // 1MB初始大小

    for (auto it = messageView.begin(); it != messageView.end(); ++it) {
        const auto& msg_view = *it;

        // 获取schema名称和topic
        const std::string& schema_name = msg_view.schema->name;
        const std::string& topic = msg_view.channel->topic;

        // 处理PointCloud2消息
        if (schema_name == "sensor_msgs/PointCloud2") {
            ROS1PointCloud2 pc2;
            if (!pc2.parse(reinterpret_cast<const uint8_t*>(msg_view.message.data),
                          msg_view.message.dataSize)) {
                std::cerr << "解析PointCloud2失败" << std::endl;
                continue;
            }

            fbb.Clear();

            // 创建Time
            foxglove::Time timestamp(pc2.header.stamp_sec, pc2.header.stamp_nsec);

            // 创建fields
            std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fb_fields;
            for (const auto& field : pc2.fields) {
                // 映射ROS数据类型到NumericType
                foxglove::NumericType numeric_type;
                switch (field.datatype) {
                    case 1:  numeric_type = foxglove::NumericType_UINT8; break;
                    case 2:  numeric_type = foxglove::NumericType_UINT16; break;
                    case 3:  numeric_type = foxglove::NumericType_UINT32; break;
                    case 7:  numeric_type = foxglove::NumericType_FLOAT32; break;
                    case 8:  numeric_type = foxglove::NumericType_FLOAT64; break;
                    default: numeric_type = foxglove::NumericType_FLOAT32; break;
                }

                auto field_name = fbb.CreateString(field.name);
                fb_fields.push_back(
                    foxglove::CreatePackedElementField(fbb, field_name, field.offset, numeric_type)
                );
            }
            auto fields_vector = fbb.CreateVector(fb_fields);

            // 创建data
            auto data_vector = fbb.CreateVector(pc2.data);

            // 创建PointCloud
            auto frame_id = fbb.CreateString(pc2.header.frame_id);
            auto pointcloud = foxglove::CreatePointCloud(
                fbb,
                &timestamp,
                frame_id,
                0,  // pose (optional)
                pc2.point_step,
                fields_vector,
                data_vector
            );

            fbb.Finish(pointcloud);

            uint64_t current_msg_time_ns = msg_view.message.logTime;
            if (last_publish_time_ns.has_value()) {
                uint64_t time_diff_ns = current_msg_time_ns - last_publish_time_ns.value();
                auto sleep_duration = std::chrono::nanoseconds(time_diff_ns);
                std::this_thread::sleep_for(sleep_duration);
            }
            last_publish_time_ns = current_msg_time_ns;

            if (!pc_publisher.publish_from_builder(fbb)) {
                std::cerr << "Failed to publish pointcloud #" << pointcloud_count << std::endl;
                continue;
            }
            
            pointcloud_count++;

            if (pointcloud_count % 100 == 0) {
                std::cout << "已 published " << pointcloud_count << " 个点云消息..." << std::endl;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // 处理CompressedImage消息
        else if (schema_name == "sensor_msgs/CompressedImage") {
            ROS1CompressedImage img;
            if (!img.parse(reinterpret_cast<const uint8_t*>(msg_view.message.data),
                          msg_view.message.dataSize)) {
                std::cerr << "解析CompressedImage失败" << std::endl;
                continue;
            }

            fbb.Clear();

            foxglove::Time timestamp(img.header.stamp_sec, img.header.stamp_nsec);
            auto frame_id = fbb.CreateString(img.header.frame_id);
            auto format = fbb.CreateString(img.format);
            auto data = fbb.CreateVector(img.data);

            auto compressed_img = foxglove::CreateCompressedImage(
                fbb,
                &timestamp,
                frame_id,
                data,
                format
            );

            fbb.Finish(compressed_img);

            uint64_t current_msg_time_ns = msg_view.message.logTime;
            if (last_publish_time_ns.has_value()) {
                uint64_t time_diff_ns = current_msg_time_ns - last_publish_time_ns.value();
                auto sleep_duration = std::chrono::nanoseconds(time_diff_ns);
                std::this_thread::sleep_for(sleep_duration);
            }
            last_publish_time_ns = current_msg_time_ns;

            if (!img_publisher.publish_from_builder(fbb)) {
                std::cerr << "Failed to publish image #" << image_count << std::endl;
                continue;
            }

            image_count++;

            if (image_count % 100 == 0) {
                std::cout << "已 published " << image_count << " 个图像消息..." << std::endl;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // 处理IMU消息
        else if (schema_name == "sensor_msgs/Imu") {
            ROS1Imu imu;
            if (!imu.parse(reinterpret_cast<const uint8_t*>(msg_view.message.data),
                          msg_view.message.dataSize)) {
                std::cerr << "解析IMU失败" << std::endl;
                continue;
            }

            fbb.Clear();

            foxglove::Time timestamp(imu.header.stamp_sec, imu.header.stamp_nsec);
            auto frame_id = fbb.CreateString(imu.header.frame_id);
            auto angular_velocity = foxglove::CreateVector3(fbb, imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
            auto linear_acceleration = foxglove::CreateVector3(fbb, imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);

            auto imu_fb = foxglove::CreateImu(fbb, &timestamp, frame_id, angular_velocity, linear_acceleration);

            fbb.Finish(imu_fb);

            uint64_t current_msg_time_ns = msg_view.message.logTime;
            if (last_publish_time_ns.has_value()) {
                uint64_t time_diff_ns = current_msg_time_ns - last_publish_time_ns.value();
                auto sleep_duration = std::chrono::nanoseconds(time_diff_ns);
                std::this_thread::sleep_for(sleep_duration);
            }
            last_publish_time_ns = current_msg_time_ns;
            
            if (!imu_publisher.publish_from_builder(fbb)) {
                std::cerr << "Failed to publish imu #" << imu_count << std::endl;
                continue;
            }
            imu_count++;

            if (imu_count % 1000 == 0) {
                std::cout << "已 published " << imu_count << " 个IMU消息..." << std::endl;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        else {
            other_count++;
        }
    }

    // 关闭文件
    reader.close();

    std::cout << "\n=== publish 完成 ===" << std::endl;
    std::cout << "publish的点云消息: " << pointcloud_count << std::endl;
    std::cout << "publish的图像消息: " << image_count << std::endl;
    std::cout << "publish的IMU消息: " << imu_count << std::endl;
    std::cout << "跳过的其他消息: " << other_count << std::endl;
}

/**
 * @brief 应用入口，读取默认文件并触发转换
 */
int main() {
    std::cout << "ROS1 MCAP publish 工具\n" << std::endl;

    const std::string input_file = "/home/ubuntu/data/meetting-c.mcap";

    convert_ros1_to_flatbuffers(input_file);

    return 0;
}
