cmake_minimum_required(VERSION 3.22)
project(slam_test LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "../slam_common/install")

find_package(slam_common REQUIRED)
find_package(OpenCV REQUIRED)

# ===================================================================
# Unit Testing with Catch2
# ===================================================================
option(BUILD_TESTS "Build the unit tests for slam_adapter" ON)

if(BUILD_TESTS)
  find_package(Catch2 REQUIRED)

  enable_testing()
  add_subdirectory(src)
endif()