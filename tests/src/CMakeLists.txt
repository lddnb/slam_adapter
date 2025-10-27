add_executable(crash_test crash_test.cpp)
target_link_libraries(crash_test PRIVATE slam_common::slam_common)

add_executable(pub_test pub_test.cpp)
target_link_libraries(pub_test PRIVATE slam_common::slam_common ${OpenCV_LIBS})
target_include_directories(pub_test PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(sub_test sub_test.cpp)
target_link_libraries(sub_test PRIVATE slam_common::slam_common ${OpenCV_LIBS})
target_include_directories(sub_test PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(pub_sub_test pub_sub_test.cpp)
target_link_libraries(pub_sub_test PRIVATE slam_common::slam_common ${OpenCV_LIBS})
target_include_directories(pub_sub_test PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(req_res_test req_res_test.cpp)
target_link_libraries(req_res_test PRIVATE slam_common::slam_common)

add_executable(callback_dispatcher_test callback_dispatcher_test.cpp)
target_link_libraries(callback_dispatcher_test PRIVATE slam_common::slam_common)

add_executable(filter_test filter_test.cpp)
target_include_directories(filter_test PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(filter_test PRIVATE slam_core::slam_core ${PCL_LIBRARIES})

include(CTest)
# Create the test executable
add_executable(point_cloud_test
    point_cloud_test.cpp
)
target_link_libraries(point_cloud_test PRIVATE Catch2::Catch2WithMain slam_core::slam_core)
add_test(NAME point_cloud_test COMMAND point_cloud_test)

add_executable(point_cloud_field_view_demo_test
    point_cloud_field_view_demo_test.cpp
)
target_link_libraries(point_cloud_field_view_demo_test PRIVATE Catch2::Catch2WithMain slam_core::slam_core)
add_test(NAME point_cloud_field_view_demo_test COMMAND point_cloud_field_view_demo_test)

# add_executable(signal_handler_test
#     signal_handler_test.cpp
# )
# target_include_directories(signal_handler_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(signal_handler_test PRIVATE slam_common::slam_common)
# add_test(NAME signal_handler_test COMMAND signal_handler_test)
# set_tests_properties(signal_handler_test PROPERTIES
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     WILL_FAIL true
# )