#include <unistd.h>

#include <cstdio>
#include <iostream>

#include <cpptrace/cpptrace.hpp>
#include <cpptrace/formatting.hpp>

auto formatter = cpptrace::formatter{}
    .addresses(cpptrace::formatter::address_mode::object)
    .snippets(true);

int main() {
    cpptrace::object_trace trace;
    while(true) {
        cpptrace::safe_object_frame frame;
        std::size_t res = fread(&frame, sizeof(frame), 1, stdin);
        if(res == 0) {
            break;
        } else if(res != 1) {
            std::cerr<<"Oops, size mismatch "<<res<<" "<<sizeof(frame)<<std::endl;
            break;
        } else {
            trace.frames.push_back(frame.resolve());
        }
    }
    formatter.print(trace.resolve());
}
