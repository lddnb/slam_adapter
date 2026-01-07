cargo build --release --package iceoryx2-ffi-c

cmake -S iceoryx2-cmake-modules -B target/ff/cmake-modules/build
cmake --install target/ff/cmake-modules/build

cmake -S iceoryx2-c -B target/ff/c/build \
      -DRUST_BUILD_ARTIFACT_PATH="$( pwd )/target/release"
cmake --build target/ff/c/build
cmake --install target/ff/c/build

cmake -S iceoryx2-bb/cxx -B target/ff/bb-cxx/build
cmake --build target/ff/bb-cxx/build
cmake --install target/ff/bb-cxx/build

cmake -S iceoryx2-cxx -B target/ff/cxx/build
cmake --build target/ff/cxx/build
cmake --install target/ff/cxx/build