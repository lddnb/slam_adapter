cmake_minimum_required(VERSION 3.22)
project(slam_common LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(spdlog REQUIRED)
find_package(cpptrace REQUIRED)
find_package(Protobuf REQUIRED)
find_package(eCAL REQUIRED)

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

# =============================================================================
# SLAM Common Library (Dynamic Library)
# =============================================================================

# 获取所有源文件
file(GLOB_RECURSE SLAM_COMMON_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
)

# 创建动态库
add_library(${PROJECT_NAME} SHARED
    ${SLAM_COMMON_SOURCES}
    ${PROTO_GENERATED_SRCS}
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# 设置动态库属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# 包含目录
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接依赖
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        cpptrace::cpptrace
        spdlog::spdlog
        protobuf::libprotobuf
        eCAL::protobuf_core
)

# 编译定义
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<BOOL:${BUILD_TESTS}>:BUILD_TESTS>
)

# =============================================================================
# Installation
# =============================================================================

# Install library and headers
install(TARGETS ${PROJECT_NAME}
    EXPORT slam_common-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install public headers
install(DIRECTORY include/slam_common DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(DIRECTORY ${PROTO_GEN_DIR}/ DESTINATION include FILES_MATCHING PATTERN "*.pb.h")

# Export configuration
install(EXPORT slam_common-targets
    FILE slam_common-targets.cmake
    NAMESPACE slam_common::
    DESTINATION lib/cmake/slam_common
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/slam_common-config-version.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/slam_common-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/slam_common-config.cmake"
    INSTALL_DESTINATION lib/cmake/slam_common
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/slam_common-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/slam_common-config-version.cmake"
    DESTINATION lib/cmake/slam_common
)

# =============================================================================
# Tools
# =============================================================================
option(BUILD_TOOLS "Build trace processing tools" ON)

if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# =============================================================================
# Package Information
# =============================================================================
set(CPACK_PACKAGE_NAME "slam_common")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SLAM Common Library")
set(CPACK_GENERATOR "DEB;RPM")

include(CPack)
