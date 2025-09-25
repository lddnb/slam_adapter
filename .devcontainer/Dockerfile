FROM ubuntu:24.04

ARG PROXY=http://127.0.0.1:7890

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
RUN sed -i 's@//.*security.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources

RUN cat /etc/apt/sources.list
RUN apt-get clean
RUN apt-get -y update --fix-missing

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

ENV http_proxy=${PROXY}
ENV https_proxy=${PROXY}
ENV HTTP_PROXY=${PROXY}
ENV HTTPS_PROXY=${PROXY}

# setup timezone
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata \
    locales \
    software-properties-common && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# install packages
RUN add-apt-repository ppa:fish-shell/release-4
RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    clangd \
    clang-format \
    curl \
    git \
    gnupg \
    pciutils \
    iputils-ping \
    iproute2 \
    libblosc-dev \
    libjemalloc-dev \
    libclang-dev \
    libc++-dev \
    binutils-dev \
    doxygen \
    expect \
    flex \
    gcc \
    gcc-multilib \
    g++ \
    g++-multilib \
    libc6-dev \
    libc6-dev-i386 \
    libc6-dev-i386-cross \
    libpython3-all-dev \
    libstdc++6-i386-cross \
    libacl1-dev \
    libelf-dev \
    libpoco-dev \
    lsb-release \
    net-tools \
    tar \
    zip \
    unzip \
    tmux \
    vim \
    wget \
    fish \
    sudo \
    xz-utils && \
    add-apt-repository universe

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    libgflags-dev \
    libunwind-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libceres-dev \
    libfmt-dev \
    libzstd-dev \
    libdwarf-dev \
    python3-pip \
    libspdlog-dev

# 清除apt缓存
RUN apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global http.proxy ${http_proxy} && \
    git config --global https.proxy ${http_proxy}

# cpptrace
RUN cd /tmp && git clone https://github.com/jeremy-rifkin/cpptrace.git && \
    cd cpptrace && \
    git checkout v1.0.4 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCPPTRACE_UNWIND_WITH_LIBUNWIND=ON && \
    make -j && \
    make install

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y

# # iceoryx2
# RUN cd /tmp && git clone https://github.com/eclipse-iceoryx/iceoryx2.git && \
#     cd iceoryx2 && \

RUN echo 'root:      ' | chpasswd
RUN echo 'ubuntu:      ' | chpasswd
USER ubuntu
WORKDIR "/home/ubuntu"

SHELL ["/bin/fish", "-c"]
RUN curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher
RUN fisher install jethrokuan/z
RUN fisher install jethrokuan/fzf
RUN fisher install jorgebucaran/replay.fish
RUN set -Ux LD_LIBRARY_PATH /usr/local/lib $LD_LIBRARY_PATH

