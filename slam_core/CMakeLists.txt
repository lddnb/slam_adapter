cmake_minimum_required(VERSION 3.22)
project(slam_core LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(manif REQUIRED)
find_package(easy_profiler REQUIRED)

# ============================================================================
# SLAM Core Library (Header-only for now)
# ============================================================================
# 获取所有源文件
file(GLOB_RECURSE SLAM_CORE_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_library(${PROJECT_NAME} SHARED ${SLAM_CORE_SOURCES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${manif_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    ${OpenCV_LIBS}
    tbb
    easy_profiler
)
target_compile_definitions(${PROJECT_NAME} PUBLIC BUILD_WITH_EASY_PROFILER)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS ${PROJECT_NAME}
    EXPORT slam_core-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/slam_core
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(EXPORT slam_core-targets
    FILE slam_core-targets.cmake
    NAMESPACE slam_core::
    DESTINATION lib/cmake/slam_core
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/slam_core-config-version.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/slam_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/slam_core-config.cmake"
    INSTALL_DESTINATION lib/cmake/slam_core
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/slam_core-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/slam_core-config-version.cmake"
    DESTINATION lib/cmake/slam_core
)
