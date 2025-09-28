cmake_minimum_required(VERSION 3.22)

# Crash file processor tool
add_executable(crash_processor crash_processor.cpp)
target_link_libraries(crash_processor cpptrace::cpptrace spdlog::spdlog)
target_compile_features(crash_processor PRIVATE cxx_std_20)

# Install the tools
install(TARGETS crash_processor
    RUNTIME DESTINATION bin
    COMPONENT tools
)