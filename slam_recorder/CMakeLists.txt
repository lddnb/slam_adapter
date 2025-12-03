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
find_package(Protobuf REQUIRED)
find_package(slam_common REQUIRED)
find_package(mcap REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# 生成并追加 Protobuf 源文件
file(GLOB_RECURSE SLAM_COMMON_PROTOS_REL RELATIVE ${PROTO_SRC_DIR} "${PROTO_SRC_DIR}/*.proto")
set(PROTO_GENERATED_SRCS)
foreach(PROTO_FILE ${SLAM_COMMON_PROTOS_REL})
    get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
    get_filename_component(PROTO_NAME_WE ${PROTO_FILE} NAME_WE)
    set(OUT_DIR ${PROTO_GEN_DIR}/${PROTO_DIR})
    file(MAKE_DIRECTORY ${OUT_DIR})
    set(GEN_SRC ${OUT_DIR}/${PROTO_NAME_WE}.pb.cc)
    set(GEN_HDR ${OUT_DIR}/${PROTO_NAME_WE}.pb.h)
    add_custom_command(
        OUTPUT ${GEN_SRC} ${GEN_HDR}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${PROTO_SRC_DIR}
            --cpp_out=${PROTO_GEN_DIR}
            ${PROTO_FILE}
        DEPENDS ${PROTO_SRC_DIR}/${PROTO_FILE}
        COMMENT "Generating protobuf for ${PROTO_FILE}"
        VERBATIM
    )
    list(APPEND PROTO_GENERATED_SRCS ${GEN_SRC})
endforeach()

add_executable(bag_tool
    src/bag_tool.cpp
)
target_include_directories(bag_tool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(bag_tool PRIVATE
    slam_common::slam_common
    mcap
    yaml-cpp
    ${OpenCV_LIBS}
)

add_executable(foxglove_bridge
    ${PROTO_GENERATED_SRCS}
    src/foxglove_bridge_main.cpp
    src/foxglove_websocket_bridge.cpp
)
target_include_directories(foxglove_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(foxglove_bridge PRIVATE
    slam_common::slam_common
    protobuf::libprotobuf
    foxglove_cpp_static
    libfoxglove.a
    yaml-cpp
    mcap
    ${OpenCV_LIBS}
)
