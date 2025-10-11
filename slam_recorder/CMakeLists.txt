cmake_minimum_required(VERSION 3.22)
project(slam_recorder LANGUAGES CXX)

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
find_package(yaml-cpp REQUIRED)

add_executable(ros_read_save src/ros_read_save.cpp)
target_include_directories(ros_read_save PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ros_read_save PRIVATE
    slam_common::slam_common
    mcap
)

add_executable(ros_read_pub src/ros_read_pub.cpp)
target_include_directories(ros_read_pub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ros_read_pub PRIVATE
    slam_common::slam_common
    mcap
)

# Foxglove WebSocket Bridge (iceoryx2 → Foxglove Studio 实时可视化)
add_executable(foxglove_bridge
    src/foxglove_websocket_bridge.cpp
    src/foxglove_bridge_main.cpp
)
target_include_directories(foxglove_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(foxglove_bridge PRIVATE
    slam_common::slam_common
    foxglove_cpp_static
    libfoxglove.a
    yaml-cpp
    mcap
)