add_executable(spdlog_crash_test spdlog_crash_test.cpp)
target_link_libraries(spdlog_crash_test slam_common::slam_common)

include(CTest)
# Create the test executable
add_executable(slam_adapter_tests
    sample_test.cpp
)
target_link_libraries(slam_adapter_tests PRIVATE Catch2::Catch2WithMain)
add_test(NAME slam_adapter_tests COMMAND slam_adapter_tests)


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