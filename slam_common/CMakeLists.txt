cmake_minimum_required(VERSION 3.22)
project(slam_common LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(iceoryx2-cmake-modules REQUIRED)
include(Iceoryx2PlatformSettings)

string(JOIN " " ICEORYX2_C_FLAGS_STRING ${ICEORYX2_C_FLAGS})
string(JOIN " " ICEORYX2_C_WARNINGS_STRING ${ICEORYX2_C_WARNINGS})
string(JOIN " " ICEORYX2_CXX_FLAGS_STRING ${ICEORYX2_CXX_FLAGS})
string(JOIN " " ICEORYX2_CXX_WARNINGS_STRING ${ICEORYX2_CXX_WARNINGS})
string(JOIN " " ICEORYX2_SANITIZER_FLAGS_STRING ${ICEORYX2_SANITIZER_FLAGS})

set(CMAKE_C_FLAGS "\
    ${CMAKE_C_FLAGS} \
    ${ICEORYX2_C_FLAGS_STRING} \
    ${ICEORYX2_C_WARNINGS_STRING}\
")
set(CMAKE_CXX_FLAGS "\
    ${CMAKE_CXX_FLAGS} \
    ${ICEORYX2_CXX_FLAGS_STRING} \
    ${ICEORYX2_CXX_WARNINGS_STRING} \
    ${ICEORYX2_SANITIZER_FLAGS_STRING} \
")

# Dependencies
find_package(iceoryx2-cxx REQUIRED)
find_package(spdlog REQUIRED)
find_package(cpptrace REQUIRED)
find_package(flatbuffers REQUIRED)

file(GLOB_RECURSE ALL_FBS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/*/*.fbs")
message(STATUS "FBS Source Files: ${ALL_FBS}")
set(ALL_FBS_HDRS)
foreach(ABS_FIL ${ALL_FBS})
  file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
  get_filename_component(DIR ${REL_FIL} DIRECTORY)
  get_filename_component(FIL_WE ${REL_FIL} NAME_WE)

  list(APPEND ALL_FBS_HDRS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}_generated.h")

  add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}_generated.h
    COMMAND  flatc --cpp -o ${PROJECT_BINARY_DIR}/${DIR} ${ABS_FIL}
    DEPENDS ${ABS_FIL}
    COMMENT "Running C++ flatbuffers compiler on ${ABS_FIL}"
  )
  message("fbs source = ${ABS_FIL}" )
  message("fbs output = ${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}_generated.h" )
endforeach()

# Create a custom target for flatbuffers generation
add_custom_target(generate_flatbuffers DEPENDS ${ALL_FBS_HDRS})

# =============================================================================
# SLAM Common Library (Dynamic Library)
# =============================================================================

# 获取所有源文件
file(GLOB_RECURSE SLAM_COMMON_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
)

# 创建动态库
add_library(${PROJECT_NAME} SHARED ${SLAM_COMMON_SOURCES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

add_dependencies(${PROJECT_NAME} generate_flatbuffers)

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
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接依赖
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        cpptrace::cpptrace
        spdlog::spdlog
        iceoryx2-cxx::static-lib-cxx
        flatbuffers::flatbuffers
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
    INCLUDES DESTINATION include
)

# Install all headers
install(DIRECTORY include/slam_common
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install generated FlatBuffers headers
if(EXISTS ${PROJECT_BINARY_DIR}/include/fbs)
    install(DIRECTORY ${PROJECT_BINARY_DIR}/include/fbs
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )
endif()

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