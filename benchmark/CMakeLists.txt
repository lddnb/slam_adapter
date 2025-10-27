cmake_minimum_required(VERSION 3.22)
project(point_cloud_benchmark LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add slam_core
list(APPEND CMAKE_PREFIX_PATH "../slam_core/install" "../thirdparty/small_gicp/install")

# Find dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(PCL 1.14 REQUIRED COMPONENTS common filters)
find_package(OpenMP)
find_package(slam_core REQUIRED)
find_package(small_gicp REQUIRED)
find_package(benchmark REQUIRED)
find_package(spdlog REQUIRED)

# ============================================================================
# Benchmark executable
# ============================================================================

add_executable(transform_benchmark src/transform_benchmark.cpp)
target_include_directories(transform_benchmark PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(transform_benchmark PRIVATE
    slam_core::slam_core
    Eigen3::Eigen
    small_gicp::small_gicp
    benchmark::benchmark
    ${PCL_LIBRARIES}
    tbb
)

add_executable(downsample_benchmark src/downsample_benchmark.cpp)
target_include_directories(downsample_benchmark PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(downsample_benchmark PRIVATE
    slam_core::slam_core
    Eigen3::Eigen
    benchmark::benchmark
    ${PCL_LIBRARIES}
    tbb
    spdlog::spdlog
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(transform_benchmark PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(transform_benchmark PRIVATE HAVE_OPENMP)
    target_link_libraries(downsample_benchmark PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(downsample_benchmark PRIVATE HAVE_OPENMP)
    message(STATUS "OpenMP enabled for benchmark")
endif()

# Print build configuration
message(STATUS "==============================================")
message(STATUS "Transform Benchmark Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  PCL: ${PCL_FOUND}")
message(STATUS "==============================================")
