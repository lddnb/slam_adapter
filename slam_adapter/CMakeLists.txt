cmake_minimum_required(VERSION 3.22)
project(slam_adapter LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress false positive warnings from fmt library in GCC 13+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    add_compile_options(-Wno-stringop-overflow)
endif()

# Add paths
list(APPEND CMAKE_PREFIX_PATH "../slam_common/install" "../slam_core/install")

# Find packages
find_package(slam_core REQUIRED)
find_package(slam_common REQUIRED)
find_package(mcap REQUIRED)

add_executable(sub_demo src/sub_demo.cpp)
target_include_directories(sub_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(sub_demo PRIVATE
    slam_core::slam_core
    slam_common::slam_common
    mcap
)
