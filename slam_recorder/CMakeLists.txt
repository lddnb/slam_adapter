cmake_minimum_required(VERSION 3.22)
project(slam_test LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add paths
list(APPEND CMAKE_PREFIX_PATH "../slam_common/install")

# Find packages
find_package(slam_common REQUIRED)
find_package(mcap REQUIRED)

# 创建可执行文件
add_executable(ros_read_save src/ros_read_save.cpp)

# 设置包含目录
target_include_directories(ros_read_save PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(ros_read_save
    slam_common::slam_common           # slam_common库 (包含FlatBuffers生成的头文件)
    mcap                                # mcap库
)

