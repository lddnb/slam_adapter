#include "slam_common/signal_handler.hpp"

void deliberately_crash() {
    int* p = nullptr;
    *p = 42;
}

int main() {
    ms_slam::slam_common::warmup_cpptrace();
    ms_slam::slam_common::InstallFailureSignalHandler();
    
    // 调用崩溃函数
    deliberately_crash();
    
    return 0;
}