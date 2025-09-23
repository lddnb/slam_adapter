cmake_minimum_required(VERSION 3.22)
project(slam_adapter LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

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

find_package(iceoryx2-cxx REQUIRED)
find_package(spdlog REQUIRED)
find_package(cpptrace REQUIRED)

add_executable(generic_demo src/generic_demo.cpp)
target_include_directories(generic_demo PRIVATE include)
target_link_libraries(generic_demo iceoryx2-cxx::static-lib-cxx spdlog::spdlog cpptrace::cpptrace)

# Installation
# install(TARGETS
#     pointcloud_publisher
#     pointcloud_subscriber
#     image_publisher
#     image_subscriber
#     demo_app
#     generic_responder
#     generic_requester
#     DESTINATION bin)
