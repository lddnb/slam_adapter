cmake_minimum_required(VERSION 3.22)
project(slam_test LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress false positive warnings from fmt library in GCC 13+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    add_compile_options(-Wno-stringop-overflow)
endif()

list(APPEND CMAKE_PREFIX_PATH "../slam_common/install" "../slam_core/install")

find_package(slam_common REQUIRED)
find_package(slam_core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(mcap REQUIRED)

# ===================================================================
# Unit Testing with Catch2
# ===================================================================
option(BUILD_TESTS "Build the unit tests for slam_adapter" ON)

if(BUILD_TESTS)
  find_package(Catch2 REQUIRED)

  enable_testing()
  add_subdirectory(src)
endif()