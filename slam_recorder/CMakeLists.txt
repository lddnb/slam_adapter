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

add_executable(bag_tool
    src/bag_tool.cpp
)
target_include_directories(bag_tool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bag_tool PRIVATE
    slam_common::slam_common
    mcap
    yaml-cpp
)

add_executable(foxglove_bridge
    src/foxglove_bridge_main.cpp
    src/foxglove_websocket_bridge.cpp
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
