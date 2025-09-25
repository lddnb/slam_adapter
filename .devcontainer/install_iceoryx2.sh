cargo build --release --package iceoryx2-ffi-c

cmake -S iceoryx2-cmake-modules -B target/ff/cmake-modules/build
cmake --install target/ff/cmake-modules/build

cmake -S iceoryx2-c -B target/ff/c/build \
      -DRUST_BUILD_ARTIFACT_PATH="$( pwd )/target/release"
cmake --build target/ff/c/build
cmake --install target/ff/c/build

git clone --depth 1 --branch v2.95.7 https://github.com/eclipse-iceoryx/iceoryx.git target/ff/iceoryx/src
cmake -S target/ff/iceoryx/src/iceoryx_platform -B target/ff/iceoryx/build/platform \
      -DCMAKE_BUILD_TYPE=Release
cmake --build target/ff/iceoryx/build/platform
cmake --install target/ff/iceoryx/build/platform

cmake -S target/ff/iceoryx/src/iceoryx_hoofs -B target/ff/iceoryx/build/hoofs \
      -DCMAKE_BUILD_TYPE=Release
cmake --build target/ff/iceoryx/build/hoofs
cmake --install target/ff/iceoryx/build/hoofs

cmake -S iceoryx2-cxx -B target/ff/cxx/build
cmake --build target/ff/cxx/build
cmake --install target/ff/cxx/build