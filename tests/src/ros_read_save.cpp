/**
 * @file ros_read_save.cpp
 * @brief ROS1 MCAP 转 FlatBuffers MCAP 保存工具
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

/**
 * @brief 将 ROS1 编码的 MCAP 文件转换为 FlatBuffers 并保存
 * @param input_file 输入 MCAP 文件路径
 * @param output_file 输出 MCAP 文件路径
 */
void convert_ros1_to_flatbuffers(const std::string& input_file, const std::string& output_file) {
    std::cout << "=== 转换ROS1 MCAP到Flatbuffers MCAP ===" << std::endl;
    std::cout << "输入文件: " << input_file << std::endl;
    std::cout << "输出文件: " << output_file << std::endl;

    // 打开输入文件
    mcap::McapReader reader;
    auto status = reader.open(input_file);
    if (!status.ok()) {
        std::cerr << "无法打开输入文件: " << status.message << std::endl;
        return;
    }

    // 创建输出writer
    mcap::McapWriter writer;
    mcap::McapWriterOptions opts("flatbuffers");
    opts.compression = mcap::Compression::Zstd;

    status = writer.open(output_file, opts);
    if (!status.ok()) {
        std::cerr << "无法创建输出文件: " << status.message << std::endl;
        reader.close();
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

            // 构建Flatbuffers消息
            flatbuffers::FlatBufferBuilder fbb(10240);

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

            // 创建/获取channel
            uint16_t channel_id;
            auto ch_it = channel_map.find(topic);
            if (ch_it == channel_map.end()) {
                channel_id = next_channel_id++;
                channel_map[topic] = channel_id;

                // 添加schema（使用嵌入的binary reflection.Schema）
                foxglove::PointCloudBinarySchema pc_bfbs;
                mcap::Schema fb_schema(
                    "foxglove.PointCloud",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(pc_bfbs.data()), pc_bfbs.size())
                );
                writer.addSchema(fb_schema);

                // 添加channel
                mcap::Channel fb_channel(topic, "flatbuffer", fb_schema.id, {});
                writer.addChannel(fb_channel);
            } else {
                channel_id = ch_it->second;
            }

            // 写入消息
            mcap::Message fb_msg;
            fb_msg.channelId = channel_id;
            fb_msg.sequence = pointcloud_count;
            fb_msg.logTime = msg_view.message.logTime;
            fb_msg.publishTime = msg_view.message.publishTime;
            fb_msg.data = reinterpret_cast<const std::byte*>(fbb.GetBufferPointer());
            fb_msg.dataSize = fbb.GetSize();

            auto write_status = writer.write(fb_msg);
            if (!write_status.ok()) {
                std::cerr << "写入点云消息失败: " << write_status.message << std::endl;
                continue;
            }
            pointcloud_count++;

            if (pointcloud_count % 100 == 0) {
                std::cout << "已转换 " << pointcloud_count << " 个点云消息..." << std::endl;
            }
        }
        // 处理CompressedImage消息
        else if (schema_name == "sensor_msgs/CompressedImage") {
            ROS1CompressedImage img;
            if (!img.parse(reinterpret_cast<const uint8_t*>(msg_view.message.data),
                          msg_view.message.dataSize)) {
                std::cerr << "解析CompressedImage失败" << std::endl;
                continue;
            }

            // 构建Flatbuffers消息
            flatbuffers::FlatBufferBuilder fbb(img.data.size() + 1024);

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

            // 创建/获取channel
            uint16_t channel_id;
            auto ch_it = channel_map.find(topic);
            if (ch_it == channel_map.end()) {
                channel_id = next_channel_id++;
                channel_map[topic] = channel_id;

                // 添加schema（使用嵌入的binary reflection.Schema）
                foxglove::CompressedImageBinarySchema img_bfbs;
                mcap::Schema fb_schema(
                    "foxglove.CompressedImage",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(img_bfbs.data()), img_bfbs.size())
                );
                writer.addSchema(fb_schema);

                // 添加channel
                mcap::Channel fb_channel(topic, "flatbuffer", fb_schema.id, {});
                writer.addChannel(fb_channel);
            } else {
                channel_id = ch_it->second;
            }

            // 写入消息
            mcap::Message fb_msg;
            fb_msg.channelId = channel_id;
            fb_msg.sequence = image_count;
            fb_msg.logTime = msg_view.message.logTime;
            fb_msg.publishTime = msg_view.message.publishTime;
            fb_msg.data = reinterpret_cast<const std::byte*>(fbb.GetBufferPointer());
            fb_msg.dataSize = fbb.GetSize();

            auto write_status = writer.write(fb_msg);
            if (!write_status.ok()) {
                std::cerr << "写入图像消息失败: " << write_status.message << std::endl;
                continue;
            }
            image_count++;

            if (image_count % 100 == 0) {
                std::cout << "已转换 " << image_count << " 个图像消息..." << std::endl;
            }
        }
        // 处理IMU消息
        else if (schema_name == "sensor_msgs/Imu") {
            ROS1Imu imu;
            if (!imu.parse(reinterpret_cast<const uint8_t*>(msg_view.message.data),
                          msg_view.message.dataSize)) {
                std::cerr << "解析IMU失败" << std::endl;
                continue;
            }

            // 构建Flatbuffers消息
            flatbuffers::FlatBufferBuilder fbb(1024);

            foxglove::Time timestamp(imu.header.stamp_sec, imu.header.stamp_nsec);
            auto frame_id = fbb.CreateString(imu.header.frame_id);
            auto angular_velocity = foxglove::CreateVector3(fbb, imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
            auto linear_acceleration = foxglove::CreateVector3(fbb, imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);

            auto imu_fb = foxglove::CreateImu(fbb, &timestamp, frame_id, angular_velocity, linear_acceleration);

            fbb.Finish(imu_fb);

            // 创建/获取channel
            uint16_t channel_id;
            auto ch_it = channel_map.find(topic);
            if (ch_it == channel_map.end()) {
                channel_id = next_channel_id++;
                channel_map[topic] = channel_id;

                foxglove::ImuBinarySchema imu_bfbs;
                mcap::Schema fb_schema(
                    "foxglove.Imu",
                    "flatbuffer",
                    std::string(reinterpret_cast<const char*>(imu_bfbs.data()), imu_bfbs.size())
                );
                writer.addSchema(fb_schema);

                // 添加channel
                mcap::Channel fb_channel(topic, "flatbuffer", fb_schema.id, {});
                writer.addChannel(fb_channel);
            } else {
                channel_id = ch_it->second;
            }

            mcap::Message fb_msg;
            fb_msg.channelId = channel_id;
            fb_msg.sequence = imu_count;
            fb_msg.logTime = msg_view.message.logTime;
            fb_msg.publishTime = msg_view.message.publishTime;
            fb_msg.data = reinterpret_cast<const std::byte*>(fbb.GetBufferPointer());
            fb_msg.dataSize = fbb.GetSize();

            auto write_status = writer.write(fb_msg);
            if (!write_status.ok()) {
                std::cerr << "写入IMU消息失败: " << write_status.message << std::endl;
                continue;
            }
            imu_count++;

            if (imu_count % 1000 == 0) {
                std::cout << "已转换 " << imu_count << " 个IMU消息..." << std::endl;
            }
        }
        else {
            other_count++;
        }
    }

    // 关闭文件
    reader.close();
    writer.close();

    std::cout << "\n=== 转换完成 ===" << std::endl;
    std::cout << "转换的点云消息: " << pointcloud_count << std::endl;
    std::cout << "转换的图像消息: " << image_count << std::endl;
    std::cout << "转换的IMU消息: " << imu_count << std::endl;
    std::cout << "跳过的其他消息: " << other_count << std::endl;
    std::cout << "输出文件: " << output_file << std::endl;
}

/**
 * @brief 应用入口，执行转换示例
 */
int main() {
    std::cout << "ROS1 MCAP 到 Flatbuffers MCAP 转换工具\n" << std::endl;

    const std::string input_file = "/home/ubuntu/data/meetting-c.mcap";
    const std::string output_file = "meetting-c-flatbuffers.mcap";

    convert_ros1_to_flatbuffers(input_file, output_file);

    return 0;
}
